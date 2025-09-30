from natsort import natsorted
import h5py
import json
from torch.utils.data import DataLoader, Dataset
import tyro
from transformers import AutoImageProcessor
from curation.suboptimal_classifier.utils import normalize_images
import os
import torch
import numpy as np
from curation.utils.view_dataset_info import apply_filter
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator
from tqdm import tqdm
import imageio
from pprint import pprint
from torchvision.transforms import functional as F
from concurrent.futures import ThreadPoolExecutor

def load_model(model_path):
    if model_path.endswith(".pth"):
        model_path = model_path
        save_dir = os.path.dirname(model_path)
    else:
        save_dir = model_path
        model_paths = natsorted([file for file in os.listdir(save_dir) if file.endswith('.pth')])
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

class Inference:
    def __init__(self, model_path):
        self.model, self.config = load_model(model_path)
        self.encoder_type = self.model.encoder_type
        self.device = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            num_gpu = torch.cuda.device_count()
            self.model = torch.nn.DataParallel(self.model, device_ids=range(num_gpu)).cuda()

        self.model.to(self.device)
        self.model.eval()
    
    def get_score(self, image):
        image = image.to(self.device)

        with torch.no_grad():
            result = self.model.forward(image=image, text=None, action=None, score=None, training=False, norm_delta_proprio=None, delta_grip_act=None)
        score = result['output'].cpu().numpy()
        
        return score

class PreprocessDataset(Dataset):
    def __init__(self, image, goal_dist, size):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.image = image
        self.size = size
        self.image = self.chunk_image(image, goal_dist)
        self.goal_dist = goal_dist
        
    def chunk_image(self, image:np.ndarray, goal_dist:np.ndarray):
        traj_len = image.shape[0] # L, H, W, C
        future_image_indices = np.arange(traj_len)[:, None] + goal_dist  # L, N
        self.future_image_indices = np.clip(future_image_indices, 0, traj_len-1) 
        self.dist = self.future_image_indices - np.arange(traj_len)[:, None]
        return image
    
    def process_image(self, image) -> torch.Tensor:
        image = torch.tensor(image) # T, H, W, C
        image = normalize_images(image)
        image = image.permute(0, 3, 1, 2) # T, C, H, W
        image = F.resize(image, [self.size, self.size])
        return image   
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image = self.image[idx]
        future_image = np.empty((len(self.goal_dist), *image.shape), dtype=image.dtype)
        for i in range(len(self.goal_dist)):
            future_image[i] = self.image[self.future_image_indices[idx, i]]
        # future_image = self.image[self.future_image_indices[idx]]
        image = self.process_image(image[None]).repeat(len(self.goal_dist), 1, 1, 1)
        future_image = self.process_image(future_image)
        image = torch.stack([image, future_image], axis=1)
        return image

class Evaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.inference = Inference(model_path)
        self.config = self.inference.config
        self.hdf5_config = self.config['hdf5_dataset_kwargs']
        
    def normal_pdf(self, x, mu=0, sigma=1):
        return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
        
    def rank_prob_to_score(self, rank_prob, goal_dist, freq, gamma=0.5):
        rank_thres = np.array(self.config['discriminator']['rank_thres'])
        all_goal_time = goal_dist/freq
        pred_rank = np.sum(rank_prob * np.arange(rank_prob.shape[-1]), axis=-1)
        score_lists = []
        for i in range(all_goal_time.shape[-1]):
            cur_goal_time = all_goal_time[:, i:i+1]
            raw_goal_rank = np.argwhere((cur_goal_time >= rank_thres[:, 0]) & (cur_goal_time < rank_thres[:, 1]))[:, 1:]
            cur_goal_rank = (cur_goal_time - rank_thres[raw_goal_rank, 0])/(rank_thres[raw_goal_rank, 1] - rank_thres[raw_goal_rank, 0]) + raw_goal_rank - 0.5
            cur_goal_rank = np.clip(cur_goal_rank, 0, rank_prob.shape[-1]-1).squeeze()
            cur_pred_rank = pred_rank[:, i:i+1].squeeze()
            cur_scores = (cur_goal_rank - cur_pred_rank) / np.where(goal_dist[:, i]<1 , 1, goal_dist[:, i]) # normalize by goal distance
            cur_scores = np.clip(cur_scores, 0, 1)

            processed_scores = np.convolve(np.squeeze(cur_scores), np.ones(int((self.goal_time[i])*freq)), 'full')
            cnt = np.convolve(np.ones_like(np.squeeze(cur_scores)), np.ones(int((self.goal_time[i])*freq)), 'full')
            
            processed_scores = (processed_scores / cnt)[:-(int(self.goal_time[i]*freq)-1)]
            for j in range(len(processed_scores)-1, 1, -1):
                processed_scores[j-1] = gamma**(1/freq) * processed_scores[j] + processed_scores[j-1]
            
            score_lists.append(processed_scores)
            
        scores = np.stack(score_lists, axis=-1)
        scores = np.mean(scores, axis=-1)
        
        return scores
    
    def smooth_score(self, score, window, mix_level):
        window = min(window, len(score))
        if window >= 1:
            cum_cnt = np.convolve(np.ones_like(score), np.ones(window), mode='same')
            cum_sum = np.convolve(score, np.ones(window), mode='same')
            smoothed_score = cum_sum / cum_cnt
        else:
            smoothed_score = score
        if mix_level > 0:
            smoothed_score = mix_level * np.mean(score) + (1-mix_level) * smoothed_score
        return smoothed_score
    
    def count_overlap_with_preintv(self, file, scores, thres):
        intv_cnt = 0
        subop_cnt = 0
        over_lap_cnt = 0
        total_cnt = 0
        for i, score in enumerate(scores):
            intv_labels = file[f'data/demo_{i}/intv_labels'][:]
            preintv_mask = intv_labels == -10
            subop_mask = score >= thres
            intv_cnt += np.sum(preintv_mask)
            subop_cnt += np.sum(subop_mask)
            subop_mask = subop_mask[:len(preintv_mask)]
            over_lap_cnt += np.sum(preintv_mask & subop_mask)
            total_cnt += len(score)
        concat_scores = np.concatenate(scores, axis=0)
        total_cnt = intv_cnt
        random_sampled = np.random.choice(np.arange(len(concat_scores)), total_cnt, replace=False)
        subop_cnt = np.sum(concat_scores[random_sampled] >= thres)
        print("intv count:", intv_cnt)
        print("subop count:", subop_cnt)
        print("overlap count:", over_lap_cnt)
        print("total count:", total_cnt)
        print("overlap/intv:", over_lap_cnt/intv_cnt)
        print("subop/total:", subop_cnt/total_cnt)

        return over_lap_cnt/intv_cnt
        
    
    def visualize(self, file, scores, percentile, visualize_num, visualize_dir, min_subop_frames=0, visualize_worst=True, smooth_window=1, mix_level=0):
        
        for i, score in enumerate(scores):
            scores[i] = np.copy(score)
            scores[i] = self.smooth_score(np.squeeze(scores[i]), smooth_window, mix_level)
            
            
        cat_scores = np.concatenate(scores, axis=0)
        thres = np.percentile(cat_scores, percentile*100)
        
        if 'round' in file['data/demo_0'].keys(): # used for sirius dataset
            round_subop_cnt = {"round_0": 0, "round_1": 0, "round_2": 0, "round_3": 0}
            round_total = {"round_0": 0, "round_1": 0, "round_2": 0, "round_3": 0}
            traj_cnt = {"round_0": 0, "round_1": 0, "round_2": 0, "round_3": 0}
            subop_traj_cnt = {"round_0": 0, "round_1": 0, "round_2": 0, "round_3": 0}
            pure_rollout_subop_cnt = {"round_0": 0, "round_1": 0, "round_2": 0, "round_3": 0}
            
            for i, score in enumerate(scores):
                demo_idx = f'demo_{i}'
                if 'round' not in file[f'data/{demo_idx}'].keys():
                    round_id = 3
                else:
                    round_id = file[f'data/{demo_idx}/round'][0]
                    if round_id == 1 and file['data'][demo_idx]['action_modes'][0] == -1:
                        round_id = 0
                round_total[f'round_{round_id}'] += len(score)
                
                if visualize_worst:
                    round_subop_cnt[f'round_{round_id}'] += sum(score >= thres)
                    subop_traj_cnt[f'round_{round_id}'] += sum(score >= thres)>min_subop_frames
                    if (file[f'data/{demo_idx}/intv_labels'][:] == 0).all():
                        pure_rollout_subop_cnt[f'round_{round_id}'] += sum(score >= thres)
                else:
                    round_subop_cnt[f'round_{round_id}'] += sum(score < (1-thres))
                    subop_traj_cnt[f'round_{round_id}'] += sum(score < (1-thres))>min_subop_frames
                    if (file[f'data/{demo_idx}/intv_labels'][:] == 0).all():
                        pure_rollout_subop_cnt[f'round_{round_id}'] += sum(score < (1-thres))
                traj_cnt[f'round_{round_id}'] += 1
            
            print('total suboptimal:', sum(cat_scores >= thres) if visualize_worst else sum(cat_scores < (1-thres)))
            print("Subop percent:")
            for key in round_subop_cnt.keys():
                if round_total[key] == 0:
                    continue
                print(key, round_subop_cnt[key]/round_total[key])
            print("Pure rollout suboptimal percent")
            for key in pure_rollout_subop_cnt.keys():
                if traj_cnt[key] == 0:
                    continue
                print(key, pure_rollout_subop_cnt[key]/round_total[key])
            pprint(round_subop_cnt)
            print("suboptimal traj count")
            pprint(subop_traj_cnt)
            print("total traj count")
            pprint(traj_cnt)
            overlaps = self.count_overlap_with_preintv(file, scores, thres)
                    
        elif 'mask' in file.keys() and 'better' in file['mask'].keys(): # used for robomimic dataset
            split_cnt = {"better": 0, "worse": 0, "okay": 0}
            frame_cnt = {"better": 0, "worse": 0, "okay": 0}
            traj_cnt = {"better": 0, "worse": 0, "okay": 0}
            subop_traj_cnt = {"better": 0, "worse": 0, "okay": 0}
            for i, score in enumerate(scores):
                demo_idx = f'demo_{i}'
                for key in split_cnt.keys():
                    demo_list = [demo.decode() for demo in file['mask'][key]]
                    if demo_idx in demo_list:
                        traj_cnt[key] += 1
                        if visualize_worst:
                            split_cnt[key] += sum(score >= thres)
                            subop_traj_cnt[key] += sum(score >= thres)>min_subop_frames
                        else:
                            split_cnt[key] += sum(score < (1-thres))
                            subop_traj_cnt[key] += sum(score < (1-thres))>min_subop_frames
                        frame_cnt[key] += len(score)
            
            print('total suboptimal:', sum(cat_scores >= thres) if visualize_worst else sum(cat_scores < (1-thres)))
            print("Subop percent:")
            for key in split_cnt.keys():
                if frame_cnt[key] == 0:
                    continue
                print(key, split_cnt[key]/frame_cnt[key])
            pprint(split_cnt)
            print("suboptimal traj count")
            pprint(subop_traj_cnt)
            overlaps = 0
        
        else: # used for sirius real
            subop_traj_cnt = 0
            for i, score in enumerate(scores):
                is_subop = score > thres
                subop_traj_cnt += sum(is_subop)>min_subop_frames
                
            print("subop_traj_cnt:", subop_traj_cnt)
            overlaps = self.count_overlap_with_preintv(file, scores, thres)
        
        if visualize_num > 0:
            traj_mask = [sum(score >= thres)>min_subop_frames for score in scores]
            masked_idx = np.argwhere(traj_mask).flatten()
            
            visualize_num = min(visualize_num, len(masked_idx))
            chosen_idx = np.random.choice(masked_idx, visualize_num, replace=False)
            
            if not os.path.exists(visualize_dir):
                os.makedirs(visualize_dir)
            
            def vis_demo(idx):
                demo_idx = f'demo_{idx}'
                
                demo = file[f'data/demo_{idx}']
                score = scores[idx]
                if visualize_worst:
                    subop_mask = np.squeeze(score >= thres)
                else:
                    subop_mask = np.squeeze(score < (1-thres))
                frames = np.array(demo['obs'][self.config['discriminator_dataset_kwargs']['image_key']])
                frames[subop_mask] = apply_filter(frames[subop_mask], (0,255,0))
                imageio.mimsave(f"{visualize_dir}/demo_{idx}.mp4", frames, fps=self.freq)
                return
            # parallelize the visualization
            futures = []
            with ThreadPoolExecutor(max_workers=1) as executor:
                for i, idx in enumerate(chosen_idx):
                    futures.append(executor.submit(vis_demo, idx))
                    
                for i, idx in tqdm(enumerate(chosen_idx)):
                    futures[i].result()
        return overlaps
                
        
    # only support latest image only discriminator
    def get_score_from_hdf5(self, data_dir, batch_size, goal_time, image_type='agentview_image', mix_level=0, visualize_num=0, visualize_percentile=0.9, visualize_dir='./vis', visualize_worst=True, save_score=False):
        file_paths = []
        self.goal_time = goal_time
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.hdf5'):
                    file_paths.append(os.path.join(root, file))  
        overlap_intv = []     
        for file_path in file_paths:
            print("Processing", file_path)
            if save_score:
                file = h5py.File(file_path, 'a')
            else:
                file = h5py.File(file_path, 'r')
            if 'env_args' in file['data'].attrs.keys():
                env_args = json.loads(file['data'].attrs['env_args'])
                name = env_args['env_name']
            else:
                name = os.path.basename(file_path).split('.')[0]
            freq = self.hdf5_config['freq']
            self.freq = freq

            goal_dist = np.array([int(time*freq) for time in goal_time])
            all_scores = []
            for i, demo_key in tqdm(enumerate(natsorted(file['data'].keys())), total=len(file['data'].keys())):
                demo = file[f'data/{demo_key}']
                preprocess_image = PreprocessDataset(demo['obs'][image_type], goal_dist, self.hdf5_config['obs_keys'][image_type])
                dataloader = DataLoader(preprocess_image, batch_size=batch_size, shuffle=False)
                traj_rank_prob = []
                for batch in dataloader:
                    rank_prob = self.inference.get_score(batch)
                    traj_rank_prob.append(rank_prob)
                traj_rank_prob = np.concatenate(traj_rank_prob, axis=0)
                scores = self.rank_prob_to_score(traj_rank_prob, preprocess_image.dist, freq)
                if save_score:
                    if 'subop_score' not in demo.keys():
                        file['data'][demo_key].create_dataset('subop_score', data=scores)
                    else:
                        file['data'][demo_key]['subop_score'][:] = scores[:].reshape(file['data'][demo_key]['subop_score'][:].shape)
                # scores = demo['subop_score'][:]
                all_scores.append(scores)
            overlap_intv.append(self.visualize(file, all_scores, visualize_percentile, visualize_num, visualize_worst=visualize_worst, visualize_dir=os.path.join(visualize_dir, name), mix_level=mix_level))
            file.close()
        print(sum(overlap_intv)/len(overlap_intv))
                    
                    
def main(
    data_dir: str, 
    model_path:str,
    goal_time:str,
    batch_size:int=256, 
    mix_level:float=0,
    visualize:int=0,
    visualize_percentile:float=0.7,
    visualize_dir:str='./vis',
    visualize_worst:bool=True,
    image_type:str='agentview_image',
    save_score:bool=False,
):          
    goal_time = [float(time) for time in goal_time.split(',')]
    evaluator = Evaluator(model_path)
    evaluator.get_score_from_hdf5(data_dir = data_dir, 
                                  batch_size = batch_size, 
                                  goal_time = goal_time, 
                                  image_type = image_type, 
                                  mix_level=mix_level,
                                  visualize_num = visualize, 
                                  visualize_percentile = visualize_percentile, 
                                  visualize_dir = visualize_dir, 
                                  visualize_worst = visualize_worst, 
                                  save_score = save_score
                                  )
                    
if __name__ == "__main__":
    tyro.cli(main)