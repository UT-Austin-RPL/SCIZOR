import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl
import numpy as np
import tyro
import itertools
from curation.utils.oxe_dataset_configs import OXE_DATASET_CONFIGS
from curation.utils.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
import imageio
import os
import cv2
from typing import Union
from natsort import natsorted

def apply_filter(frame, color):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    color_mask = np.zeros_like(frame)
    color_mask[:] = color
    frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

class TrajLoader:
    def __init__(self, name:str, data_dir:str, split:str="train"):
        self.name = name
        self.data_dir = data_dir
        self.new_dataset = None
        self.builder = tfds.builder(name, data_dir=data_dir)
        self.dataset = self.builder.as_dataset(split=split,
                                               decoders={"steps": tfds.decode.SkipDecoding()},
                                               shuffle_files=False, 
                                               read_config=tfds.ReadConfig(add_tfds_id=True, num_parallel_calls_for_interleave_files=1, interleave_cycle_length=1, interleave_block_length=1)
                                        
                                               )

    
    def print_info(self):
        print(f"Dataset Name: {self.name}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Number of demos: {len(self.dataset)}")
        
        # dataset_iterator = self.dataset.iterator()
        len_list = []
        proprio_diff_list = []
        transform = OXE_STANDARDIZATION_TRANSFORMS[self.name]
        self.dataset.prefetch(tf.data.AUTOTUNE)
        for i,demo in enumerate(self.dataset):
            print(f"Processing {i+1}/{len(self.dataset)}", end='\r')
            demo = demo['steps']
            demo = transform(demo)
            traj_len = tf.shape(demo["is_first"])[0]
            len_list.append(traj_len.numpy().item())
            proprio = demo['observation']["proprio"].numpy()
            proprio_diff = np.diff(proprio, axis=0)
            abs_diff = np.abs(proprio_diff).sum(axis=-1)
            mean_diff = np.mean(abs_diff)
            proprio_diff_list.append(mean_diff)
            if len(len_list) % min(100,len(self.dataset)-1) == 0:
                print(f"First 1000 demos: Mean {np.mean(len_list)}, Std {np.std(len_list)}, Max {np.max(len_list)}, Min {np.min(len_list)}, Mean Proprio Diff {np.mean(proprio_diff_list)}")
                break
        return self.dataset
    
    def visualize(self, index:list[Union[int, bytes, tuple]], save_dir:str='./video', save_instruction:bool=False):
        """ 
        Visualize the demos from the tfds dataset with the given indices
        Args:
            index: list of indices of the demos to visualize, it can be a list of integers or a list of tuples with the first element as the index and the second element as the tfds_id
                    If the tfds_id:byte is provided, it will be checked against the tfds_id in the dataset.
            save_dir: directory to save the videos
        """
        def save_video(save_dir, video_name, images, demo, save_instruction):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imageio.mimsave(f"{save_dir}/{video_name}.mp4", images, fps=10)
            print(f"Saved {video_name}.mp4", end='\r')
            if save_instruction:
                demo['steps'] = transform(demo['steps'])
                language_instruction = demo['steps']['language_instruction'][0].numpy().decode()
                with open(f"{save_dir}/{video_name}.txt", 'w') as f:
                    f.write(f"{language_instruction}")
        
        obs_key = OXE_DATASET_CONFIGS[self.name]['image_obs_keys']['primary']
        transform = OXE_STANDARDIZATION_TRANSFORMS[self.name]
            
        if len(index) == 0:
            return
        
        if isinstance(index[0], tuple):
            is_byte = [isinstance(index[0][i], bytes)  for i in range(len(index[0]))]
            byte_idx = np.argwhere(is_byte).flatten()
            has_byte = np.any(is_byte)
            is_str = [isinstance(index[0][i], str) for i in range(len(index[0]))]
            str_idx = np.argwhere(is_str).flatten()
            has_str = np.any(is_str)
            if has_byte:
                index = natsorted(index, key=lambda x: x[byte_idx[0]].decode())
            elif has_str:
                index = natsorted(index, key=lambda x: x[str_idx[0]])
            else:
                index = natsorted(index, key=lambda x: x[0])
        elif isinstance(index[0], bytes):
            index = natsorted(index, key=lambda x: x.decode())
        else:
            index = natsorted(index)
                    
        def process_tfds_id(tfds_id):
            if isinstance(tfds_id, bytes):
                split = tfds_id.split(b'---')
            elif isinstance(tfds_id, str):
                split = tfds_id.split('---')
            if len(split) == 2:
                tfds_id, frame = split
                frame = int(frame.decode())
            else:
                frame = None
            return tfds_id, frame
        if isinstance(index[0], bytes):
            tfds_idxs = [process_tfds_id(idx)[0] for idx in index]
        for i, demo in enumerate(self.dataset):
            first = True
            demo['tfds_id'] = demo['tfds_id'].numpy()
            print(demo['tfds_id'], end='\r')
            while len(index) > 0:
                idx_to_pop = 0
                frame, tfds_id, ind, frames = None, None, None, None
                if isinstance(index[0], tuple):
                    for j in range(len(index[0])):
                        if isinstance(index[0][j], int) or isinstance(index[0][j], np.int32):
                            # ind = index[0][j]
                            pass
                        elif isinstance(index[0][j], bytes) or isinstance(index[0][j], str):
                            tfds_id, frame = process_tfds_id(index[0][j])
                        elif isinstance(index[0][j], list):
                            frames = index[0][j]
                elif isinstance(index[0], int):
                    ind = index[0]
                    tfds_id, frame = None, None
                elif isinstance(index[0], bytes):
                    try:
                        idx_to_pop = tfds_idxs.index(demo['tfds_id'])
                    except:
                        idx_to_pop = None
                        break
                    tfds_id, frame = process_tfds_id(index[idx_to_pop])
                    tfds_idxs.pop(idx_to_pop)
                    ind = None
                # print(demo['tfds_id'], tfds_id)
                if i != ind and demo['tfds_id'] != tfds_id and demo['tfds_id'].decode() != tfds_id:
                    break
                else:
                    index.pop(idx_to_pop)

                if first:
                    demo['image'] = tf.map_fn(tf.image.decode_jpeg, demo['steps']['observation'][obs_key], dtype=tf.uint8).numpy()
                
                if tfds_id is None:
                    tfds_id = demo['tfds_id'].decode()
                else:
                    if isinstance(tfds_id, bytes):
                        tfds_id = tfds_id.decode()
                
                if demo['tfds_id'].decode()!=tfds_id:
                    print(f"tfds_id not matched: {demo['tfds_id'].decode()} != {tfds_id} for index {ind}")
                else:
                    if frame is not None:
                        assert frames is None
                        start = max(frame-2, 0)
                        end = min(frame+4, len(demo['image']))
                        images = demo['image'][start:end]
                        images[2] = apply_filter(images[2], (0,255,0))
                        video_name = f"{tfds_id}---{frame}"
                        save_video(save_dir, video_name, images, demo, save_instruction)
                    else:
                        images = demo['image']
                        video_name = tfds_id + f"---{frames[0]}-{frames[-1]}"
                        if frames is not None:
                            for i in range(len(frames)):
                                if isinstance(frames[i], int):
                                    images[frames[i]] = apply_filter(images[frames[i]], (0,255,0))
                                elif isinstance(frames[i], tuple):
                                    cur_frame, rgb = frames[i]
                                    images[cur_frame] = apply_filter(images[cur_frame], rgb)
                        save_video(save_dir, video_name, images, demo, save_instruction)
                    
            if len(index) == 0:
                break
        
    
if __name__ == "__main__":
    loader = tyro.cli(TrajLoader)
    loader.print_info()
    
    