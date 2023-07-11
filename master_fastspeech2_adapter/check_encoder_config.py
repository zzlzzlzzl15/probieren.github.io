<<<<<<< HEAD
from torch.autograd import gradcheck
import torch
import torch.nn as nn
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from Utility.storage_config import MODELS_DIR, PREPROCESSING_DIR
import os



path_to_checkpoint=os.path.join(MODELS_DIR, "FastSpeech2_BitFit", "best.pt")
check_dict = torch.load(path_to_checkpoint)
model = FastSpeech2()
model.load_state_dict(check_dict["model"])

#print(model.encoder.__dict__)
"""for n,p in model.encoder.named_parameters():
    print(n[-8:])
"""
"""
possible_keys = ["self_attn"]
for k in possible_keys:
    if k in model.encoder.__dict__:
        print(getattr(model.encoder, k)) # +2 for embedding layer and last layer
"""
"""
prefix = model.__class__.__name__
for name, module in model.named_modules():
    try:
        items = module._modules.items()
        assert(len(items))

    except:
        print(name)

"""

"""
for name_1,par in model.named_parameters():
    if "weight" in name_1:
        #print(name[-7:])

        for name_model,par_model in model.named_modules():
            if name_1[:-7]==name_model and name_1[:16] == "encoder.encoders":
                print(name_model)
"""

"""
for name,parameters in model.encoder.named_parameters():
    print(name,':',parameters.size())
    
"""
def cal_sparsity_pen(total_layers):
    sparsity_pen = [1.25e-7]*total_layers
    return sparsity_pen

def get_layer_idx_from_module(self, n: str) -> int:
    # get layer index based on module name
    num_layer = 0
    for k in self.encoder.named_modules():
        num_layer += 1
    return num_layer


def get_encoder_base_modules(self, return_names: bool = False):
    if self._parametrized:
        check_fn = lambda m: hasattr(m, "parametrizations")
    else:
        check_fn = lambda m: len(m._parameters) > 0
    return [(n, m) if return_names else m for n, m in self.encoder.named_modules() if check_fn(m)]


def total_layers(model) -> int:
    num_layer = 0
    for k in model.encoder.named_modules():
        num_layer += 1
    return num_layer

def cal_module_pen():
    for module_name, base_module in get_encoder_base_modules(return_names=True):
        layer_idx = get_layer_idx_from_module(module_name)
        sparsity_pen_final = cal_sparsity_pen(layer_idx)
        module_pen = 0.
        for par_list in list(base_module.parametrizations.values()):
            for a in par_list[0].alpha_weights:
                module_pen += get_l0_norm_term(a, log_ratio)
        l0_pen += (module_pen * sparsity_pen_final)

=======
from torch.autograd import gradcheck
import torch
import torch.nn as nn
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from Utility.storage_config import MODELS_DIR, PREPROCESSING_DIR
import os



path_to_checkpoint=os.path.join(MODELS_DIR, "FastSpeech2_BitFit", "best.pt")
check_dict = torch.load(path_to_checkpoint)
model = FastSpeech2()
model.load_state_dict(check_dict["model"])

#print(model.encoder.__dict__)
"""for n,p in model.encoder.named_parameters():
    print(n[-8:])
"""
"""
possible_keys = ["self_attn"]
for k in possible_keys:
    if k in model.encoder.__dict__:
        print(getattr(model.encoder, k)) # +2 for embedding layer and last layer
"""
"""
prefix = model.__class__.__name__
for name, module in model.named_modules():
    try:
        items = module._modules.items()
        assert(len(items))

    except:
        print(name)

"""

"""
for name_1,par in model.named_parameters():
    if "weight" in name_1:
        #print(name[-7:])

        for name_model,par_model in model.named_modules():
            if name_1[:-7]==name_model and name_1[:16] == "encoder.encoders":
                print(name_model)
"""

"""
for name,parameters in model.encoder.named_parameters():
    print(name,':',parameters.size())
    
"""
def cal_sparsity_pen(total_layers):
    sparsity_pen = [1.25e-7]*total_layers
    return sparsity_pen

def get_layer_idx_from_module(self, n: str) -> int:
    # get layer index based on module name
    num_layer = 0
    for k in self.encoder.named_modules():
        num_layer += 1
    return num_layer


def get_encoder_base_modules(self, return_names: bool = False):
    if self._parametrized:
        check_fn = lambda m: hasattr(m, "parametrizations")
    else:
        check_fn = lambda m: len(m._parameters) > 0
    return [(n, m) if return_names else m for n, m in self.encoder.named_modules() if check_fn(m)]


def total_layers(model) -> int:
    num_layer = 0
    for k in model.encoder.named_modules():
        num_layer += 1
    return num_layer

def cal_module_pen():
    for module_name, base_module in get_encoder_base_modules(return_names=True):
        layer_idx = get_layer_idx_from_module(module_name)
        sparsity_pen_final = cal_sparsity_pen(layer_idx)
        module_pen = 0.
        for par_list in list(base_module.parametrizations.values()):
            for a in par_list[0].alpha_weights:
                module_pen += get_l0_norm_term(a, log_ratio)
        l0_pen += (module_pen * sparsity_pen_final)

>>>>>>> a6dfef13c4529bfe33644a990df8499785b6aa11
print(total_layers(model))