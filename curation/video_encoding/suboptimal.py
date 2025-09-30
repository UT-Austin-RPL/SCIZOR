import torch
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from curation.suboptimal_classifier.dataset.rlds.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS, OXE_DATASET_CONTROL_FREQUENCY
from curation.suboptimal_classifier.dataset.rlds.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from tqdm import tqdm
import tyro
import os
import pickle
from threading import Thread, Lock
from time import sleep
from typing import Union
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator
from curation.suboptimal_classifier.utils import normalize_images, process_char_array_for_dino
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.chdir("../suboptimal_classifier")
import json
import natsort
from octo.octo.data.utils.data_utils import get_dataset_statistics, tree_map, normalize_action_and_proprio
from octo.octo.utils.spec import ModuleSpec
from curation.utils.view_dataset_info import apply_filter
import imageio
import ray
from matplotlib import pyplot as plt
import dlimp as dl
import concurrent
from utils import get_statistics


class ModelBuffer:
    def __init__(self, max_buffer_size=1024, batch_size=64) -> None:
        self.buffer = {'images':[], 'text':[], 'action':[], 'norm_delta_proprio':[], 'delta_grip_act':[]}
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size        
        self.no_more_data = False
        self.buffer_lock = Lock()
        
    def __len__(self):
        self.buffer_lock.acquire()
        len_buffer = len(self.buffer['images'])
        self.buffer_lock.release()
        return len_buffer
        
    
    def append(self, image:np.ndarray, action:np.ndarray, text:Union[str, np.ndarray]):#, norm_delta_proprio:np.ndarray, delta_grip_act:np.ndarray):
        assert type(text) in [str, np.ndarray]
        assert type(image) == np.ndarray
        while len(self.buffer['images']) >= self.max_buffer_size:
            sleep(0.1)
        self.buffer_lock.acquire()
        self.buffer['images'].append(image)
        self.buffer['action'].append(action)
        self.buffer['text'].append(text)
        # self.buffer['norm_delta_proprio'].append(norm_delta_proprio)
        # self.buffer['delta_grip_act'].append(delta_grip_act)
        self.buffer_lock.release()
        
    def extend(self, images:list[np.ndarray], action:list[np.ndarray], texts:list[Union[str, np.ndarray]]):#, norm_delta_proprio:list[np.ndarray], delta_grip_act:list[np.ndarray]):
        assert len(images) == len(texts) == len(action)
        while len(self.buffer['images']) >= self.max_buffer_size:
            sleep(0.1)
        self.buffer_lock.acquire()
        self.buffer['images'].extend(images)
        self.buffer['action'].extend(action)
        self.buffer['text'].extend(texts)
        # self.buffer['norm_delta_proprio'].extend(norm_delta_proprio)
        # self.buffer['delta_grip_act'].extend(delta_grip_act)
        self.buffer_lock.release()
                        
        
    def get_batch(self):
        if len(self.buffer['images']) > 0:
            self.buffer_lock.acquire()
            if len(self.buffer['images']) > self.batch_size:
                inputs = {'images': self.buffer['images'][:self.batch_size], 
                          'action': self.buffer['action'][:self.batch_size],
                          'text': self.buffer['text'][:self.batch_size],
                        #   'norm_delta_proprio': self.buffer['norm_delta_proprio'][:self.batch_size],
                        #   'delta_grip_act': self.buffer['delta_grip_act'][:self.batch_size]
                          }
                self.buffer['images'] = self.buffer['images'][self.batch_size:]
                self.buffer['action'] = self.buffer['action'][self.batch_size:]
                self.buffer['text'] = self.buffer['text'][self.batch_size:]
                # self.buffer['norm_delta_proprio'] = self.buffer['norm_delta_proprio'][self.batch_size:]
                # self.buffer['delta_grip_act'] = self.buffer['delta_grip_act'][self.batch_size:]
            else:
                inputs = {'images': [self.buffer['images'].pop(0)], 
                          'action': [self.buffer['action'].pop(0)],
                          'text': [self.buffer['text'].pop(0)],
                        #   'norm_delta_proprio': [self.buffer['norm_delta_proprio'].pop(0)],
                        #   'delta_grip_act': [self.buffer['delta_grip_act'].pop(0)]
                          }
            self.buffer_lock.release()
            return inputs
        else:
            return None
        
    
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
    
    
def get_statistics(name, builder):
    dataset = dl.DLataset.from_rlds(builder, split='all', shuffle=False)
    dataset = dataset.traj_map(OXE_STANDARDIZATION_TRANSFORMS[name])
    transform_spec = ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])
    proprio_obs_key = 'proprio'
    filter_function = ()
    
    dataset_statistics = get_dataset_statistics(
            dataset,
            hash_dependencies=(
                str(builder.info),
                str(proprio_obs_key),
                ModuleSpec.to_string(transform_spec).replace("curation.utils","octo.data.oxe")
                if transform_spec is not None
                else "",
                *map(ModuleSpec.to_string, filter_function)
            ),
            save_dir=builder.data_dir,
            force_recompute=False,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)
    # action_statistics = dataset_statistics['action']
    return dataset_statistics    

def chunk_obs_act(demo, action_history, image_history, future_action, goal_dist, goal_first=False):
    traj_len = tf.shape(demo['action'])[0]

    action_history_indices = tf.range(traj_len)[:, None] + tf.range(
        -action_history + 1, future_action + 1
    )
    image_history_indices = tf.range(traj_len)[:, None] + tf.range(
        -image_history + 1, 1
    )
    
    if goal_dist is not None:
        future_image_indices = tf.range(traj_len)[:, None] + tf.constant([goal_dist])
        future_image_indices = tf.clip_by_value(future_image_indices, 0, traj_len - 1)
        future_image = tf.gather(demo['images'], future_image_indices)
        demo['goal_dist'] = future_image_indices - tf.range(traj_len)[:, None]
    
    action_history_indices = tf.clip_by_value(action_history_indices, 0, traj_len - 1)
    image_history_indices = tf.clip_by_value(image_history_indices, 0, traj_len - 1)
    
    demo['images'] = tf.gather(demo['images'], image_history_indices)
    if goal_dist is not None:
        demo['images'] = tf.repeat(demo['images'][:, None], len(goal_dist), axis=1)
        future_image = future_image[:, :, None]
        if not goal_first:
            demo['images'] = tf.concat([demo['images'], future_image], axis=2)
        else:
            demo['images'] = tf.concat([future_image, demo['images']], axis=1)
    demo['action'] = tf.gather(demo['action'], action_history_indices)
    # demo['norm_delta_proprio'] = tf.gather(demo['norm_delta_proprio'], action_history_indices)
    # demo['delta_grip_act'] = tf.gather(demo['delta_grip_act'], action_history_indices)
    
    return demo

num_gpus = torch.cuda.device_count()
class Preprocessor:
    def __init__(self, encoder_type, no_text_input):
        self.device = torch.device('cpu')
        self.encoder_type = encoder_type
        self.no_text_input = no_text_input
        if self.encoder_type != "groundingdino":
            from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
            self.text_encoder = AutoModel.from_pretrained('distilbert/distilbert-base-uncased').to(self.device)
    
    def process_batch(self, batch):
        image, action, language_instruction, norm_delta_proprio, delta_grip_act = batch['images'], batch['action'], batch['text'], None, None #, batch['norm_delta_proprio'], batch['delta_grip_act']
        image = torch.tensor(np.stack(image, axis=0))
        action = torch.tensor(np.stack(action, axis=0))
        # delta_grip_act = torch.tensor(np.stack(delta_grip_act, axis=0))
        # norm_delta_proprio = torch.tensor(np.stack(norm_delta_proprio, axis=0))
        image_shape = image.shape
        # image = torch.tensor(image)
        language_instruction = process_char_array_for_dino(np.array(language_instruction))
        
        if self.encoder_type != "groundingdino":
            image = image.view(-1, *image.shape[-3:])
            image = self.processor(image, return_tensors='pt')['pixel_values']
            image = image.view(*image_shape[:-3], *image.shape[1:])
            if not self.no_text_input:
                language_instruction = self.tokenizer(language_instruction, return_tensors='pt', padding=True, truncation=True).to(self.device)
                language_instruction = self.text_encoder(**language_instruction)["last_hidden_state"]
            else:
                language_instruction = language_instruction
        else:
            image = normalize_images(image, batched=True)
            image = image.permute(0, 1, 4, 2, 3) # B, T, C, H, W
        return image, action, language_instruction, norm_delta_proprio, delta_grip_act    

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
    
    def get_score(self, batch):
        image, action, language_instruction, norm_delta_proprio, delta_grip_act = batch
        
        norm_delta_proprio = torch.zeros_like(action[...,0])
        delta_grip_act = torch.zeros_like(action[...,0])
        
        image, action, norm_delta_proprio, delta_grip_act = image.to(self.device), action.to(self.device), norm_delta_proprio.to(self.device), delta_grip_act.to(self.device)

        with torch.no_grad():
            result = self.model.forward(image, language_instruction, action, score=None, training=False, norm_delta_proprio=norm_delta_proprio, delta_grip_act=delta_grip_act)
        score = result['output'].cpu().numpy()
        
        return score
    
from multiprocessing import Process, Pipe, Queue


def run_inference_server(model_path, pipe, score_queue):
    inference_server = Inference(model_path)
    while True:
        batch = pipe.recv()
        if batch == 'Preprocess Done':
            pipe.send('Encode Done')
            break
        score = inference_server.get_score(batch)
        score_queue.put(score)
    pipe.recv()
    pipe.send('Done')
    
    
def preprocess(batch):
    global preprocessor
    batch = preprocessor.process_batch(batch)
    return batch

class FeatureExtractor:
    def __init__(self, model_path, batch_size=64, goal_time=None, num_cpus=8):
        self.pipe, child_pipe = Pipe()
        self.score_queue = Queue()
        self.inference_server = Process(target=run_inference_server, args=(model_path, child_pipe, self.score_queue))
        self.inference_server.start()
        self.model, self.config = load_model(model_path)
        self.encoder_type = self.model.encoder_type
        global preprocessor
        preprocessor = Preprocessor(self.encoder_type, self.model.no_text_input)
        self.image_history = self.model.window_size
        self.action_history = self.model.action_horizon
        self.future_action = self.model.future_action
        self.future_image = self.model.future_image
        self.goal_time = goal_time
        if self.future_image:
            goal_time_stat = self.config['dataset_kwargs']['traj_transform_kwargs']['goal_relabeling_kwargs']
            self.goal_time_stat = {'max': goal_time_stat['max_goal_time_diff'], 'min': goal_time_stat['min_goal_time_diff']}
        else:
            self.goal_time_stat = None
        self.goal_relabeling_strategy = self.config['dataset_kwargs']['traj_transform_kwargs']['goal_relabeling_strategy']
        # assert self.window_size == 1, "Only support window size 1 for now"

        self.buffer = ModelBuffer(batch_size=batch_size)
        self.model_outputs = {'score':[], }
        self.goal_dist_list = []
        self.meta = {"traj_meta":dict()}
        self.batch_size = batch_size   
        self.preprocess_future = []
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
        
        
        self.encode_thread = Thread(target=self._encode_thread_fn)
        self.encode_thread.start()     
        self.image_to_visualize = []
    
    def get_score(self, batch):
        batch = self.preprocessor.process_batch(batch)
        return batch
    
    def _encode_thread_fn(self):
        while (not self.buffer.no_more_data) or len(self.buffer) > 0 or len(self.preprocess_future) > 0:
            
            while len(self.buffer) != 0 and len(self.preprocess_future) < 16:
                batch = self.buffer.get_batch()
                self.preprocess_future.append((self.pool.submit(preprocess, batch), batch))
            
            if len(self.preprocess_future) == 0:
                continue
            cur_future, cur_batch = self.preprocess_future.pop(0)
            try:
                result = cur_future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                print("TimeoutError, resubmitting the batch for processing")
                self.preprocess_future.insert(0, (self.pool.submit(preprocess, cur_batch), cur_batch))
                continue
                
            self.pipe.send(result)
            
        

    def get_score_from_tfds(self, dataset_name, data_dir, split='train', visualize=0, visualize_interval=10, image_type='primary', no_assign=False, no_discount=False):
        self.no_assign = no_assign
        self.no_discount = no_discount
        self.idx_cnt = 0
        total_len = 0
        
        with tf.device('/CPU:0'):
            builder = tfds.builder(dataset_name, data_dir=data_dir)
            dataset = builder.as_dataset(split=split, decoders={"steps": tfds.decode.SkipDecoding()}, shuffle_files=False, read_config=tfds.ReadConfig(add_tfds_id=True))

            obs_key = OXE_DATASET_CONFIGS[dataset_name]['image_obs_keys'][image_type]
            self.freq = OXE_DATASET_CONTROL_FREQUENCY[dataset_name]
            goal_first = self.goal_relabeling_strategy == 'custom_uniform_v2'
            if self.future_image:
                if self.goal_relabeling_strategy != 'custom_uniform_v2':
                    if self.goal_time is not None:
                        goal_dist = [int(time * self.freq) for time in self.goal_time]
                    else:
                        assert self.goal_time_stat['max'], "Please provide the goal time"
                        goal_dist = int(self.goal_time_stat['max'] * self.freq) 
                else:
                    goal_dist = 1000000 # a large number to take the last frame as the goal
            else:
                goal_dist = None
            dataset_statistics = get_statistics(dataset_name, builder)
            
            def std_transform(x):
                steps = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](x['steps'])
                steps['tfds_id'] = x['tfds_id']
                return steps
            
            def filter_non_zero_length(x):
                return tf.shape(x['action'])[0] > 0
            
            def get_dict(x):
                steps = x
                traj_len = tf.shape(steps["action"])[0]
                    
                traj = {
                    'images': tf.map_fn(tf.image.decode_jpeg, steps['observation'][obs_key], dtype=tf.uint8)[:traj_len, ...] if obs_key is not None else\
                        tf.zeros((traj_len, 256, 256, 3), dtype=tf.uint8),
                    'id': x['tfds_id'],
                    'action': steps['action'],
                    'language_instruction': steps['language_instruction'][:traj_len],
                    'proprio': steps['observation']['proprio'][:traj_len],
                }
                xyzrpy = (traj['action'][:, :-1] - dataset_statistics['action']['mean'][:-1]) / dataset_statistics['action']['std'][:-1]
                traj['action'] = tf.concat([xyzrpy, traj['action'][:, -1:]], axis=-1)
                # traj['proprio'] = (traj['proprio'] - dataset_statistics['proprio']['mean']) / dataset_statistics['proprio']['std']
                # traj['proprio'] = traj['proprio'][:, -1:] # only use the gripper proprio
                # traj['norm_delta_proprio'] = tf.linalg.norm(traj['proprio'][1:] - traj['proprio'][:-1], axis=-1)
                # traj['norm_delta_proprio'] = tf.concat([traj['norm_delta_proprio'], traj['norm_delta_proprio'][-1:]], axis=0)
                # traj['norm_delta_proprio'] = tf.pad(traj['norm_delta_proprio'], [[1, 0]], constant_values=0)
                # grip_action = steps['action'][:, -1]
                # pad_grip_action = tf.pad(grip_action, [[1, 0]], constant_values=1)
                # traj['delta_grip_act'] = pad_grip_action[1:] - pad_grip_action[:-1]
                traj = chunk_obs_act(traj, self.action_history, self.image_history, self.future_action, goal_dist, goal_first=goal_first)
                return traj
            length = len(dataset)
            
            dataset = dataset.map(std_transform, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.filter(filter_non_zero_length)
            dataset=dataset.map(get_dict, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
            self.store_list = {'id':[]}
            
            for i,traj in enumerate(tqdm(dataset, total=length)):
                assert traj['action'].shape[0] == len(traj['images']) == len(traj['language_instruction'])
                traj_len = traj['action'].shape[0]
                
                for key in traj.keys():
                    traj[key] = traj[key].numpy()
                # if traj['id'].decode() not in ['fractal20220817_data-train.tfrecord-00994-of-01024__6', 'fractal20220817_data-train.tfrecord-00987-of-01024__15', 'fractal20220817_data-train.tfrecord-00986-of-01024__43']: continue
                self.meta['traj_meta'][traj['id']] = {'start': self.idx_cnt, 'end': self.idx_cnt + traj_len}
                self.meta['traj_meta'][traj['id']]['language_instruction'] = traj['language_instruction'][0]
                
                self.buffer.extend(
                    list(traj['images']), 
                    list(traj['action']),
                    list(traj['language_instruction']),
                    # list(traj['norm_delta_proprio']),
                    # list(traj['delta_grip_act'])
                )
                if self.future_image:
                    self.goal_dist_list.append(traj['goal_dist'])
                self.idx_cnt += len(traj['images'])
                if visualize and i % visualize_interval == 0:
                    visualize -= 1
                    self.image_to_visualize.append((traj['id'], traj['images']))
                
                while not self.score_queue.empty():
                    score = self.score_queue.get(timeout=30)
                    self.model_outputs['score'].append(score)
                    total_len += score.shape[0]
                
                # for key in self.store_list.keys():
                #     self.store_list[key].append(traj[key])
            self.buffer.no_more_data = True
            print(f"Waiting for the encoding thread to finish, {len(self.buffer)} frames left in the buffer")
            
            # wait for len(buffer) to be 0
            buffer_len = len(self.buffer)
            tqdm_bar = tqdm(total=buffer_len)
            while len(self.buffer) > 0:
                tqdm_bar.n = buffer_len - len(self.buffer)
                tqdm_bar.update(0)
                sleep(1.0)
            self.encode_thread.join()
            
            self.pipe.send('Preprocess Done')
            done = self.pipe.recv()
            assert done == 'Encode Done'
            
            # Get all score from the queue
            while total_len < self.idx_cnt:
                score = self.score_queue.get(timeout=30)
                self.model_outputs['score'].append(score)
                total_len += score.shape[0]
                
            # Signal the inference server to die
            self.pipe.send('Done')
            print("Encoding thread finished")
    
    def save_output(self, path, visualize_threshold=0.5, gamma=0.5):
        path = os.path.expanduser(path)
        print(f"Saving the output to {path}")
        
        scores = np.concatenate(self.model_outputs['score'], axis=0)
        if self.future_image:
            goal_dists = np.concatenate(self.goal_dist_list, axis=0)
            if self.config['discriminator']['loss_fn_type'] != 'cross_entropy':
                goal_score = 1 - goal_dists/self.freq/self.goal_time_stat['max']
                scores = scores - goal_score
            elif self.config['discriminator']['head_type'] == 'regression':
                scores = 1/(1 + np.exp(-scores))
            elif self.config['discriminator']['head_type'] == 'rank':
                rank_thres = np.array(self.config['discriminator']['rank_thres'])
                all_goal_time = goal_dists/self.freq
                pred_rank = np.sum(scores * np.arange(scores.shape[-1]), axis=-1)
                score_lists = []
                for i in range(all_goal_time.shape[-1]):
                    cur_goal_time = all_goal_time[:, i:i+1]
                    raw_goal_rank = np.argwhere((cur_goal_time >= rank_thres[:, 0]) & (cur_goal_time < rank_thres[:, 1]))[:, 1:]
                    cur_goal_rank = (cur_goal_time - rank_thres[raw_goal_rank, 0])/(rank_thres[raw_goal_rank, 1] - rank_thres[raw_goal_rank, 0]) + raw_goal_rank - 0.5
                    cur_goal_rank = np.clip(cur_goal_rank, 0, scores.shape[-1]-1).squeeze()
                    cur_pred_rank = pred_rank[:, i:i+1].squeeze()
                    cur_scores = (cur_goal_rank - cur_pred_rank) / np.where(goal_dists[:, i]<1 , 1, goal_dists[:, i]) # normalize by goal distance
                    if not self.no_assign:
                        processed_scores = np.convolve(np.squeeze(cur_scores), np.ones(int((self.goal_time[i])*self.freq)), 'full')
                        cnt = np.convolve(np.ones_like(np.squeeze(cur_scores)), np.ones(int((self.goal_time[i])*self.freq)), 'full')
                        
                        processed_scores = (processed_scores / cnt)[:-(int(self.goal_time[i]*self.freq)-1)]
                        if not self.no_discount:
                            for j in range(len(processed_scores)-1, 1, -1):
                                processed_scores[j-1] = gamma**(1/self.freq) * processed_scores[j] + processed_scores[j-1]
                    else:
                        processed_scores = cur_scores
                    
                    score_lists.append(processed_scores)
                    
                scores = np.stack(score_lists, axis=-1)
                scores = scores.clip(0, 1)
                scores = np.mean(scores, axis=-1)
        elif self.goal_relabeling_strategy == 'uniform_action':
            scores = scores[:, :, 1]
            
        scores = scores.clip(0, 1)
        total_frames = scores.shape[0]
        
        print("Mean score: ", np.mean(scores))
        
        np.save(f'{path}/scores.npy', scores)
        
        for i,traj_id in enumerate(self.meta['traj_meta'].keys()):
            start = self.meta['traj_meta'][traj_id]['start']
            end = self.meta['traj_meta'][traj_id]['end']
            
        with open(f'{path}/traj_meta.pkl', 'wb') as f:
            pickle.dump(self.meta, f)
            
        print("Saved the output")
        
        if len(self.image_to_visualize) == 0:
            return
        
        visualize_dir = os.path.join(path, 'visualize')
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        
        for meta in self.meta['traj_meta'].values():
            meta['start'] = int(meta['start'])
            meta['end'] = int(meta['end'])
            scores = scores.flatten()
            pad_scores = np.pad(scores[meta['start']:meta['end']], (3,2), mode='edge')
            scores[meta['start']:meta['end']] = np.convolve(pad_scores, np.ones(5)/5, mode='same')[3:-2]
            
            
        visualize_threshold = np.percentile(scores, visualize_threshold*100)
        for i, (tfds_id, images) in enumerate(self.image_to_visualize):
            start, end = self.meta['traj_meta'][tfds_id]['start'], self.meta['traj_meta'][tfds_id]['end']
            language_instruction = self.meta['traj_meta'][tfds_id]['language_instruction']
            score = scores[start:end].flatten()
            
            plt.plot(score)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(visualize_dir, f"{tfds_id.decode()}_{language_instruction}_{visualize_threshold}.png"))
            plt.cla()
            
            if self.goal_relabeling_strategy != 'custom_uniform_v2':
                subop_mask = score > visualize_threshold
            else:
                subop_mask = score[2*self.freq:] < score[:-2*self.freq]
                subop_mask = np.concatenate([subop_mask, [[False]]*self.freq*2])
                
                de_sigmoid_score = -np.log(1/score-1)
                plt.plot(de_sigmoid_score)
                plt.savefig(os.path.join(visualize_dir, f"{tfds_id.decode()}_{language_instruction}_{visualize_threshold}_de_sigmoid.png"))
                plt.cla()
            images = images[:, 0, 0] # discard future frames
            subop_mask = subop_mask.flatten()
            if subop_mask.sum() == 0:
                pass
            else:
                images[subop_mask] = apply_filter(images[subop_mask], (255,0,0))
            imageio.mimsave(os.path.join(visualize_dir, f"{tfds_id.decode()}_{language_instruction}_{visualize_threshold}.mp4"), images, fps=20)
            
            
            
        
        
    
def main(name: str, 
         data_dir: str, 
         model_path:str,
         output_dir:str=None,
         split:str="train[:95%]", 
         batch_size:int=256, 
         visualize:int=0,
         visualize_thres:float=0.5,
         visualize_interval:int=1,
         device_id:str=None,
         num_cpus:int=8,
         goal_time:str=None,
         image_type:str='primary',
         no_assign:bool=False,
         no_discount:bool=False,
         ):
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    if goal_time is not None:
        goal_time = [float(time) for time in goal_time.split(',')]
    feature_extractor = FeatureExtractor(model_path, batch_size=batch_size, goal_time=goal_time, num_cpus=num_cpus)
    feature_extractor.get_score_from_tfds(name, data_dir, split, visualize, visualize_interval, image_type=image_type, no_assign=no_assign, no_discount=no_discount)
    
    if output_dir is not None:
        out_dir = f'{output_dir}/{name}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        feature_extractor.save_output(out_dir, visualize_threshold=visualize_thres)
    
if __name__ == '__main__':
    tyro.cli(main)