import torch 
import numpy as np
import wandb
from flax.traverse_util import flatten_dict
import datetime
from collections import OrderedDict
from concurrent.futures import Future

class Torch_Preallocater:
    def __init__(self, ratio, devices):
        self.preallocate_tensor = self.preallocate(ratio, devices)
        
    def preallocate(self, ratio, devices):
        total_memory = [torch.cuda.get_device_properties(device).total_memory for device in devices]
        preallocate_tensor = [torch.zeros(int(ratio*total_memory[i]), device=device, dtype=torch.uint8) for i, device in enumerate(devices)]
        return preallocate_tensor
    
    def free(self):
        del self.preallocate_tensor
        torch.cuda.empty_cache()
        
        
def process_char_array_for_dino(arr: np.ndarray):
    arr = arr.astype('S')
    arr = np.char.strip(arr)
    end_with_period = np.char.endswith(arr, b'.')
    arr[~end_with_period] = np.char.add(arr[~end_with_period], b'.')
    arr = [desc.decode('utf-8') for desc in arr]
    return arr

def normalize_images(images: torch.Tensor, batched: bool = True):
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    size = (1, 1, 1, 3) if batched else (1, 1, 3)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(*size)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(*size)
    images = (images - mean) / std
    return images

def wandb_init(config, name, debug):
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if debug else None,
        **config.wandb,
    )
    return wandb_id

def wandb_log(info, step):
    wandb.log(flatten_dict(info, sep="/"), step=step)
    
class TrainInfo:
    def __init__(self):
        self.info = {}
        self.buffer = []
    
    def update_dict(self, info):
        def update_list(info, key, value):
            if key not in info:
                info[key] = []
            info[key].append(value)
            
        for key, value in info.items():

            if isinstance(value, float):
                update_list(self.info, key, value)
                
            elif isinstance(value, dict):
                if key not in self.info:
                    self.info[key] = {}
                for subkey, subvalue in value.items():
                    update_list(self.info[key], subkey, subvalue)
                    
    def update(self, info):
        if isinstance(info, dict):
            self.update_dict(info)
        elif isinstance(info, Future):
            self.buffer.append(info)
            
    def get_average(self):
        while len(self.buffer) > 0:
            info = self.buffer.pop(0).result()
            self.update(info)
            
        average = OrderedDict()
        for key, value in self.info.items():
            if isinstance(value, list):
                average[key] = sum(value) / len(value)
                self.info[key] = []
            elif isinstance(value, dict):
                average[key] = {}
                for subkey, subvalue in value.items():
                    average[key][subkey] = sum(subvalue) / len(subvalue)
                    self.info[key][subkey] = []
        
        sorted_average = OrderedDict(sorted(average.items()))            
        return sorted_average