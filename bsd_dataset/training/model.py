from ..models import *

def load(model, **kwargs):
    if(model == "LinReg"):
        return LinReg(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "AvgPool"):
        return AvgPool(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "SRCNN"):
        return SRCNN(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "UNet"):
        return UNet(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"])
    if(model == "PerceiverIO"):
        return Perceiver(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"], model_config = kwargs["model_config"])
    if(model == "Transformer"):
        return Transformer(input_shape = kwargs["input_shape"], target_shape = kwargs["target_shape"], model_config = kwargs["model_config"])
    raise Exception(f"Model {model} is not supported")