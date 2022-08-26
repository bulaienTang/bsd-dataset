from collections import defaultdict
from itertools import product
from pathlib import Path
import os
import re
import tarfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import xarray as xr

import bsd_dataset
from bsd_dataset.regions import Region
from bsd_dataset.datasets.dataset_utils import get_lon_mask
from bsd_dataset.datasets.download_utils import (
    DatasetRequest,
    CDSAPIRequestBuilder,
    download_urls, 
    multidownload_from_cds,     
)

class BSDD(torch.utils.data.Dataset):

    def __init__(
        self,
        X: torch.Tensor,
        X_meta: Dict[str, Any],
        Y: torch.Tensor,
        Y_meta: Dict[str, Any],
        transform: Callable = None,
        target_transform: Callable = None,
        device: Union[str, torch.device] = 'cpu'
    ):
        self.X = X
        self.X_meta = X_meta
        self.Y = Y
        self.Y_meta = Y_meta
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:        
        x = self.X[idx]
        y = self.Y[idx]

        x_lats = self.X_meta['lat']
        x_lons = self.X_meta['lon']
        x_lats = np.repeat(x_lats[:, np.newaxis], len(x_lons), axis=1)
        x_lons = np.repeat(x_lons[np.newaxis, :], len(x_lats), axis=0)
        x_lats = torch.tensor(x_lats).to(self.device)
        x_lons = torch.tensor(x_lons).to(self.device)
        
        y_lats = self.Y_meta['lat']
        y_lons = self.Y_meta['lon']
        y_lats = np.repeat(y_lats[:, np.newaxis], len(y_lons), axis=1)
        y_lons = np.repeat(y_lons[np.newaxis, :], len(y_lats), axis=0)
        y_lats = torch.tensor(y_lats).to(self.device)
        y_lons = torch.tensor(y_lons).to(self.device)

        mask = torch.isnan(y).to(self.device)

        info = {
            'x_lat': x_lats,
            'x_lon': x_lons,
            'y_lat': y_lats,
            'y_lon': y_lons,
            'y_mask': mask,
            'channels': self.X_meta['channels']
        }

        if self.transform is not None:
            x = self.transform(x, info)
        if self.target_transform is not None:
            y = self.target_transform(y, info)
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y, info

class BSDDBuilder:

    def __init__(
        self, 
        input_datasets: List[DatasetRequest],
        target_dataset: DatasetRequest,
        train_region: Region,
        val_region: Region,
        test_region: Region,
        train_dates: Tuple[str, str],
        val_dates: Tuple[str, str],
        test_dates: Tuple[str, str],
        root: Path,
        device: Union[str, torch.device] = 'cpu'
    ):       
        # Validate datasets
        for ds_req in input_datasets:
            if ds_req.dataset not in bsd_dataset.input_datasets:
                raise ValueError(
                    f'Requested dataset {ds_req.dataset} is unrecognized.\n'
                    f'Must be one of {bsd_dataset.input_datasets}.'
                )
        if target_dataset.dataset not in bsd_dataset.target_datasets:
            raise ValueError(
                f'The target dataset "{target_dataset.dataset}" is unrecognized.\n'
                f'Must be one of {bsd_dataset.target_datasets}.'
            )

        # Validate dates
        if np.datetime64(train_dates[0]) > np.datetime64(train_dates[1]):
            raise ValueError(
                'End of training period should come after the start'
            )
        if np.datetime64(val_dates[0]) > np.datetime64(val_dates[1]):
            raise ValueError(
                'End of validation period should come after the start'
            )
        if np.datetime64(test_dates[0]) > np.datetime64(test_dates[1]):
            raise ValueError(
                'End of testing period should come after the start'
            )

        # Save arguments
        self.input_datasets = input_datasets
        self.target_dataset = target_dataset
        self.train_region = train_region
        self.val_region = val_region
        self.test_region = test_region
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
        self.device = torch.device(device)

        # Check that root exists
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.root = root

        # Build parameters
        self.built_download_requests = False

    def prepare_download_requests(self) -> Self:
        cds_api_requests, cds_dstdirs = [], []
        builder = CDSAPIRequestBuilder()
        input_urls, input_dstdirs = [], []
        for ds_req in self.input_datasets:
            if ds_req.is_cds_req():
                cds_api_reqs = builder.build(
                    self.root,
                    ds_req, 
                    self.train_dates, 
                    self.val_dates, 
                    self.test_dates
                )
                for car in cds_api_reqs:
                    direc = car.output.parent / car.output.name.split('.')[1]
                    cds_dstdirs.append(direc)
                cds_api_requests.extend(cds_api_reqs)
            if ds_req.dataset == 'gmted2010':
                input_urls.append(self.get_gmted2010_url(ds_req))
                direc = self.root / 'gmted2010'
                direc.mkdir(parents=True, exist_ok=True)
                input_dstdirs.append(direc)
    
        target_urls, target_dstdir = [], Path()
        if self.target_dataset.dataset == 'chirps':
            chirps_urls = self.get_chirps_urls(self.target_dataset)
            target_urls.extend(chirps_urls)
            target_dstdir = self.root / 'chirps'
        if self.target_dataset.dataset == 'persiann-cdr':
            persianncdr_urls = self.get_persianncdr_urls()
            target_urls.extend(persianncdr_urls)
            target_dstdir = self.root / 'persiann-cdr'
        target_dstdir.mkdir(parents=True, exist_ok=True)
                
        self.cds_api_requests = cds_api_requests
        self.cds_dstdirs = list(set(cds_dstdirs))
        self.input_urls = input_urls
        self.input_dstdirs = input_dstdirs
        self.target_urls = target_urls
        self.target_dstdir = target_dstdir
        self.built_download_requests = True

        return self
       
    def download(self) -> Self:
        if not self.built_download_requests:
            print('ERROR: download requests not yet built')
            return
        print(
            '========================= WARNING =========================\n'
            'If requesting a lot of data (several GB) from CDS, CDS may\n'
            'take a long while to prepare the data for you. You can\n'
            'check on the status of your request at this link:\n'
            'https://cds.climate.copernicus.eu/cdsapp#!/yourrequests.\n'
            '==========================================================='
        )
        target_dstdirs = [self.target_dstdir] * len(self.target_urls)
        download_urls(
            self.input_urls + self.target_urls, 
            self.input_dstdirs + target_dstdirs,
            n_workers=5
        )
        multidownload_from_cds(self.cds_api_requests, n_workers=5)        
        return self
        
    def extract(self) -> Self:
        if not self.built_download_requests:
            print('ERROR: download requests not yet built')
            return
        
        splits = ['train', 'val', 'test']

        # Get the CDS data
        self.extract_cds_zipped()
        cds_data = defaultdict(dict)
        for spl, direc in product(splits, self.cds_dstdirs):
            data_src = direc.name
            this_data = self.extract_cds_data(spl, direc)
            for var_name, (data, lat, lon) in this_data:
                cds_data[spl][f'{data_src}:{var_name}:data'] = data
                cds_data[spl]['lat'] = lat
                cds_data[spl]['lon'] = lon

        # Get the GMTED2010 data
        gmted_present = False
        for ds in self.input_datasets:            
            if ds.dataset == 'gmted2010':
                gmted_present = True
        
        if gmted_present:
            gmted2010_data = dict()
            for spl in splits:
                gmted2010_data[spl] = self.extract_gmted2010_data(spl, self.root / 'gmted2010')

        # Save separately to save room on disk, scaling and concatenation can happen later
        for spl in splits:
            with open(self.root / f'{spl}_x.npz', 'wb') as f:
                if gmted_present:
                    np.savez(f, **cds_data[spl], gmted2010=gmted2010_data[spl])
                else:
                    np.savez(f, **cds_data[spl])

        # Get target data
        target_data = defaultdict(dict)
        dstdir = self.target_dstdir
        
        if self.target_dataset.dataset == 'chirps':
            for spl in splits:
                chirps_data, chirps_lats, chirps_lons = self.extract_chirps_data(spl, dstdir)
                target_data[spl]['target'] = chirps_data
                target_data[spl]['lat'] = chirps_lats
                target_data[spl]['lon'] = chirps_lons
        
        if self.target_dataset.dataset == 'persiann-cdr':
            for spl in splits:
                persiann_data, persiann_lats, persiann_lons = self.extract_persiann_data(spl, dstdir)
                target_data[spl]['target'] = persiann_data
                target_data[spl]['lat'] = persiann_lats
                target_data[spl]['lon'] = persiann_lons
        
        # Save to disk
        for spl in splits:
            with open(self.root / f'{spl}_y.npz', 'wb') as f:
                np.savez(f, **target_data[spl])

        return self

    def get_split(self, split: str, transform: Callable = None, target_transform: Callable = None) -> BSDD:
        splits = ['train', 'val', 'test']
        if split not in splits:
            raise ValueError(f'Split {split} not recognized\nMust be of {splits}')
        X, X_meta, Y, Y_meta = self.load_XY(f'../era-eu-1.4-persiann-0.25/{split}_x.npy', f'../era-eu-1.4-persiann-0.25/{split}_y.npy')
        dataset = BSDD(X, X_meta, Y, Y_meta, transform, target_transform, self.device)
        return dataset

    def load_XY(self, Xfile, Yfile) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(self.root / Xfile, 'rb') as f:
            npzfile = np.load(f)
            arrs = []
            channels = []
            for key in npzfile.files:
                if key not in ('gtmed2010', 'lat', 'lon'):
                    arrs.append(npzfile[key])
                    channels.append(key)
            if 'gmted2010' in npzfile.files:
                # TODO @jasonjewik: verify each array in the file has the same number of days
                n_days = arrs[0].shape[0]
                gmted2010 = npzfile['gmted2010']
                gmted2010 = np.expand_dims(gmted2010, 0)
                gmted2010 = np.repeat(gmted2010, n_days, axis=0)
                arrs.append(gmted2010)
            X = np.stack(arrs, axis=1)  # days x channel x lon x lat
            lats = npzfile['lat']
            lons = npzfile['lon']
            X_meta = {'lat': lats, 'lon': lons, 'channels': channels}
        with open(self.root / Yfile, 'rb') as f:
            npzfile = np.load(f)
            Y = npzfile['target']
            lats = npzfile['lat']
            lons = npzfile['lon']
            Y_meta = {'lat': lats, 'lon': lons}
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return (X, X_meta, Y, Y_meta)

    def get_gmted2010_url(self, ds_req: DatasetRequest) -> str:
        resolutions = [0.0625, 0.125, 0.25, 0.5, 0.75, 1]
        res = getattr(ds_req, 'resolution', None)
        if res not in resolutions:
            raise AttributeError(
                'gmted2010 dataset has invalid resolution\n'
                f'available resolutions are {resolutions}'
            )
        if res == 0.0625:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n015_00625deg.nc'
        elif res == 0.125:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n030_0125deg.nc'
        elif res == 0.25:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0250deg.nc'
        elif res == 0.5:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n060_0500deg.nc'
        elif res == 0.75:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n180_0750deg.nc'
        elif res == 1:
            url = 'https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/GMTED2010_15n240_1000deg.nc'
        return url

    def get_chirps_urls(self, ds_req: DatasetRequest) -> List[str]:
        resolutions = {0.05: '05', 0.25: '25'}
        res = getattr(ds_req, 'resolution', None)
        str_res = resolutions.get(res, None)
        if str_res is None:
            raise AttributeError(
                'chirps dataset has invalid resolution,'
                f' available resolutions are {resolutions}'
            )
        
        train_years = [int(d.split('-')[0]) for d in self.train_dates]
        val_years = [int(d.split('-')[0]) for d in self.val_dates]
        test_years = [int(d.split('-')[0]) for d in self.test_dates]
        urls = []
        
        for a, b in [train_years, val_years, test_years]:
            if not (a in range(1981, 2022) and b in range(1981, 2022)):
                raise ValueError(
                    f'requested dates ({a}, {b}) are out of range for CHIRPS,'
                    ' which must be in 1981-2021'
                )
            urls.extend([
                'ftp://anonymous@ftp.chc.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'
                f'global_daily/netcdf/p{str_res}/chirps-v2.0.{year}.days_p{str_res}.nc'
                for year in range(a, b+1)
            ])
        
        return urls

    def get_persianncdr_urls(self) -> List[str]:
        urls = []
        persiann_start = np.datetime64('1983-01-01')
        persiann_end = np.datetime64('2021-12-31')

        for start, end in [self.train_dates, self.val_dates, self.test_dates]:
            start_time = np.datetime64(start)
            end_time = np.datetime64(end)
            if persiann_start <= start_time <= end_time <= persiann_end:
                pass
            else:
                raise ValueError(
                    f'requested dates ({start}, {end}) are out of range for PERSIANN-CDR,'
                    ' which must be in 1983-2021'
                )
        
        train_dates = pd.date_range(*self.train_dates)
        val_dates = pd.date_range(*self.val_dates)
        test_dates = pd.date_range(*self.test_dates)
        years = [d.year for d in train_dates]
        years.extend([d.year for d in val_dates])
        years.extend([d.year for d in test_dates])
        years = sorted(set(years))
        index_urls = [
            f'https://www.ncei.noaa.gov/data/precipitation-persiann/access/{y}/'
            for y in years
        ]
        dstdirec = self.root / 'tmp'
        dstdirec.mkdir(parents=True, exist_ok=True)
        dsts = [dstdirec / f'persiann.{y}.html' for y in years]
        download_urls(index_urls, dsts)

        urls = []
        for dst in dsts:
            with open(dst, 'r') as f:
                for line in f:
                    match = re.search('>PERSIANN-CDR_.*.nc<', line)
                    if match:
                        fname = match.group(0)[1:-1]
                        year = fname.split('_')[2][:4]
                        urls.append(
                            'https://www.ncei.noaa.gov/data/precipitation-persiann'
                            f'/access/{year}/{fname}'
                        )

        return urls

    def extract_cds_zipped(self) -> None:
        for req in self.cds_api_requests:
            src = req.output
            dir_path = src.parent / src.name.split('.')[1]
            if src.suffix == '.tgz':
                with tarfile.open(src) as tar:        
                    tar.extractall(path=dir_path)
            if src.suffix == '.zip':
                with ZipFile(src) as zipfile:
                    zipfile.extractall(path=dir_path)

    def extract_cds_data(self, split: str, src_dir: Path) -> List[Tuple[str, np.ndarray]]:
        region = getattr(self, f'{split}_region')
        dates = getattr(self, f'{split}_dates')
        lons = region.get_longitudes(360)
        lats = region.get_latitudes()
        var_names = []
        for fname in os.listdir(src_dir):
            if Path(fname).suffix == '.nc':
                var_names.append(fname.split('_')[0])
        result = []
        for vn in var_names:
            ds = xr.open_mfdataset(str(src_dir / f'{vn}*.nc'))
            # If the top-left has a longitude greater than the bottom-left, we have to 
            # use a mask (e.g., TL: 350, BR: 40 does not mean we want 40 to 350 degrees),
            # but we want to wrap the other way around the globe from 350 to 40 degrees.
            if lons[0] > lons[1]:
                lon_slice = get_lon_mask(ds.lon, lons)
            else:
                lon_slice = slice(*lons)
            xdata = getattr(ds, vn).sel(
                time=slice(*dates),
                lat=slice(*sorted(lats)),
                lon=lon_slice)
            npdata = xdata.values  # time x lat x lon
            this_lats = xdata.lat.values
            this_lons = xdata.lon.values
            # Convert lons from [0, 360] to [-180, 180]
            this_lons -= 180
            result.append((vn, (npdata, this_lats, this_lons)))
        return result

    def extract_gmted2010_data(self, split: str, src: Path) -> np.ndarray:
        region = getattr(self, f'{split}_region')
        lons = region.get_longitudes(180)
        lats = region.get_latitudes()
        ds = xr.open_mfdataset(str(src / '*.nc'))
        xdata = ds.elevation.sel(
            latitude=slice(*sorted(lats)), 
            longitude=slice(*lons))
        npdata = xdata.values.T  # lon x lat
        return npdata
            
    def extract_chirps_data(self, split: str, src: Path) -> np.ndarray:
        dates = getattr(self, f'{split}_dates')
        region = getattr(self, f'{split}_region')
        lons = region.get_longitudes(180)
        lats = region.get_latitudes()
        ds = xr.open_mfdataset(str(src / '*.nc'))
        xdata = ds.precip.sel(
            time=slice(*dates),
            latitude=slice(*sorted(lats)), 
            longitude=slice(*lons))
        # Drop leap days since CDS datasets don't have those
        xdata = xdata.sel(time=~((xdata.time.dt.month == 2) & (xdata.time.dt.day == 29)))
        npdata = xdata.values  # time x lat x lon
        this_lats = xdata.latitude.values
        this_lons = xdata.longitude.values
        return (npdata, this_lats, this_lons)
    
    def extract_persiann_data(self, split: str, src: Path) -> np.ndarray:
        dates = getattr(self, f'{split}_dates')
        region = getattr(self, f'{split}_region')
        lons = region.get_longitudes(360)
        lats = region.get_latitudes()
        ds = xr.open_mfdataset(str(src / '*.nc'))
        # Put latitudes into ascending order
        ds = ds.reindex(lat=ds.lat[::-1])
        # If the top-left has a longitude greater than the bottom-left, we have to 
        # use a mask (e.g., TL: 350, BR: 40 does not mean we want 40 to 350 degrees),
        # but we want to wrap the other way around the globe from 350 to 40 degrees.
        if lons[0] > lons[1]:
            lon_slice = get_lon_mask(ds.lon, lons)
        else:
            lon_slice = slice(*lons)
        xdata = ds.precipitation.sel(
            time=slice(*dates),
            lat=slice(*sorted(lats)),
            lon=lon_slice)
        # Drop leap days 
        xdata = xdata.sel(time=~((xdata.time.dt.month == 2) & (xdata.time.dt.day == 29)))
        npdata = xdata.values  # time x lon x lat
        npdata = np.moveaxis(npdata, 1, 2)  # time x lat x lon
        this_lats = xdata.lat.values
        this_lons = xdata.lon.values
        return (npdata, this_lats, this_lons)
