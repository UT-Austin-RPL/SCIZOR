import natsort.natsort
from curation.suboptimal_classifier.dataset.dataset import Hdf5SubopDataset
from torch.utils.data import DataLoader
import torch
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator
from curation.suboptimal_classifier.utils import TrainInfo
import os
import json
import natsort
from pprint import pprint
import tyro
from tqdm import tqdm
import torch.multiprocessing as mp

def evaluate_dataloader(model:Discriminator, dataloader:DataLoader, metric:str):
    eval_info = TrainInfo()
    target_score = 0.0 if metric == 'rand' else 1.0
    with torch.no_grad():
        for i,batch in enumerate(tqdm(dataloader)):
            image, action, language_instruction, norm_delta_proprio, delta_grip_act, label = batch
            image, action, norm_delta_proprio, delta_grip_act, label = image.cuda(), action.cuda(), norm_delta_proprio.cuda(), delta_grip_act.cuda(), label.cuda()
            # language_instruction = language_instruction[0]
            # score = torch.ones(action.shape[0], dtype=torch.float32, device=action.device) * target_score
            score = label
            result = model.forward(image, language_instruction, action, norm_delta_proprio=norm_delta_proprio, delta_grip_act=delta_grip_act, score=score, training=False)
            eval_info.update(result['metrics'])
    return eval_info.get_average()

def get_dataloader_for_eval(h5df_path:str, batch_size:int=32, window_size:int=1, action_horizon:int=1, future_action:int=0, future_image:bool=False, zero_score_only:bool=False, one_score_only:bool=False):
    datasets = Hdf5SubopDataset(h5df_path, window_size, action_horizon, future_action=future_action, future_image=future_image, zero_score_only=zero_score_only, one_score_only=one_score_only)
    for dataset_name, metric, sub_dataset in datasets.all_dataset():
        if mp.get_start_method() != 'fork':
            num_workers = 0
        else:
            num_workers = 8
        print(f"Using {num_workers} workers")
            
        dataloader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        yield dataset_name, metric, dataloader
        

def evaluate_dataset(h5df_path:str, model, batch_size:int=32, zero_score_only:bool=False, one_score_only:bool=False):
    results = {}
    dataloader_iter = get_dataloader_for_eval(h5df_path, batch_size, model.window_size, model.action_horizon, model.future_action, model.future_image, zero_score_only, one_score_only)
    for dataset_name, metric, dataloader in dataloader_iter:
        if dataset_name not in results:
            results[dataset_name] = {}
        print("-"*30)
        print(f"Evaluating {dataset_name} {metric}")
        results[dataset_name][metric] = evaluate_dataloader(model, dataloader, metric)
        
    return results

def load_model(model_path):
    if model_path.endswith(".pth"):
        model_path = model_path
        save_dir = os.path.dirname(model_path)
    else:
        save_dir = model_path
        model_paths = natsort.natsorted([file for file in os.listdir(save_dir) if file.endswith('.pth')])
        model_path = os.path.join(save_dir, model_paths[-1])
        
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = Discriminator(**config['discriminator'])
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print("Removing the prefix 'module.' from the keys")
        state_dict = torch.load(model_path)
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace('module.', '')] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
    return model, config
        
def main(model_path:str, hdf5_path:str, batch_size:int=32, zero_score_only:bool=False, one_score_only:bool=False):
    model, config = load_model(model_path)
    model = model.cuda()
    
    results = evaluate_dataset(hdf5_path, model, batch_size, zero_score_only, one_score_only)
    
    pprint(results)
    
if __name__ == '__main__':
    tyro.cli(main)
    