import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .misc import parsing_layers, get_model_config

def model_multigpu(model, gpus, meta=None, model_name=None):
    assert meta is not None or model_name is not None, "at least one of 'meta' or 'model_name' argument must not None"
    
    if meta is None:
        meta = get_model_config(model_name)

    layers, pre_layers, post_layers = parsing_layers(model=model, meta=meta)
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(gpus[0])
    
    for post_layer in post_layers:
        post_layer = post_layer.to(gpus[0])
    
    model.lm_head = model.lm_head.to(gpus[0])

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            for key in kwargs:
                if hasattr(kwargs[key], 'device') and kwargs[key].device != self.dev:
                    kwargs[key] = kwargs[key].to(self.dev)
            tmp = self.module(*inp, **kwargs)
            return tmp

    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers) - 1):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))
    layers[-1] = MoveModule(layers[-1].to(gpus[0]))

    model.gpus = gpus

def get_hfmodel(model_name_or_path: str,
                dtype='auto',
                device_map='cpu',
                trust_remote_code=False,
                ):
    
    # for faster model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code, 
    )
    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model
