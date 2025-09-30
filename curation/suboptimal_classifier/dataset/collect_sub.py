import tensorflow_datasets as tfds
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from curation.utils.view_dataset_info import apply_filter
from curation.utils.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from curation.utils.oxe_dataset_configs import OXE_DATASET_CONFIGS, OXE_DATASET_CONTROL_FREQUENCY
from octo.octo.data.utils.data_utils import get_dataset_statistics, tree_map, normalize_action_and_proprio
from octo.octo.utils.spec import ModuleSpec

import os
import tyro
import h5py
import cv2
import time
import random
from pprint import pprint
import imageio
import pickle

def chunk_window(demo, window_size, future_step):
    traj_len = tf.shape(demo['action'])[0]

    history_indices = tf.range(traj_len)[:, None] + tf.range(
        -window_size + 1, 1
    )
    history_indices = tf.clip_by_value(history_indices, 0, traj_len - 1)
    future_indices = tf.range(traj_len)[:, None] + tf.range(1, window_size + 1)
    future_indices = tf.clip_by_value(future_indices, 0, traj_len - 1)
    future_image_indices = tf.range(traj_len) + tf.constant(future_step)
    future_image_indices = tf.clip_by_value(future_image_indices, 0, traj_len - 1)
    demo['future_image'] = tf.gather(demo['image'], future_image_indices)
    demo['image'] = tf.gather(demo['image'], history_indices)
    demo['future_action'] = tf.gather(demo['action'], future_indices)
    demo['action'] = tf.gather(demo['action'], history_indices)
    demo['norm_delta_proprio'] = tf.gather(demo['norm_delta_proprio'], history_indices)
    demo['delta_grip_act'] = tf.gather(demo['delta_grip_act'], history_indices)
    
    return demo

def get_datasets(name, data_dir, window_size=2, future_step_scale=1, split='val'):
    builder = tfds.builder(name, data_dir=data_dir)
    if split == 'val':
        if "val" not in builder.info.splits:
            split = "train[95%:]"
        else:
            split = "val"
    elif split == 'train':
        split = "train[:95%]"
    else:
        split = split
        
    dataset = builder.as_dataset(split=split,
                                 decoders={'steps': tfds.decode.SkipDecoding()},
                                 shuffle_files=False, 
                                 read_config=tfds.ReadConfig(add_tfds_id=True),
                                 )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    transform_spec = ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])
    std_transform = ModuleSpec.instantiate(transform_spec)
    image_key = OXE_DATASET_CONFIGS[name]['image_obs_keys']['primary']
    dataset_freq = OXE_DATASET_CONTROL_FREQUENCY[name]
    proprio_obs_key = "proprio"
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
    def demo_std_transform(demo):
        demo['steps'] = std_transform(demo['steps'])
        return demo
        
        
    def transform(demo):
        steps = demo['steps']
        traj_len = tf.shape(steps['action'])[0]
        result_dict = {}
        result_dict['image'] = tf.map_fn(tf.image.decode_jpeg, steps['observation'][image_key], dtype=tf.uint8)
        except_grip_proprio = (steps['observation']['proprio'][:, :-1]-dataset_statistics['proprio']['mean'][:-1])/(dataset_statistics['proprio']['std'][:-1] + 1e-8)
        result_dict['proprio'] = tf.concat([except_grip_proprio, steps['observation']['proprio'][:, -1:]], axis=-1)
        result_dict['norm_delta_proprio'] = tf.norm(result_dict['proprio'][1:, -1:] - result_dict['proprio'][:-1, -1:], axis=-1)# only use the gripper proprio
        result_dict['norm_delta_proprio'] = tf.concat([result_dict['norm_delta_proprio'], result_dict['norm_delta_proprio'][-1:]], axis=0)
        grip_action = steps['action'][:, -1]
        xyz_rpy = steps['action'][:, :-1]
        xyz_rpy = (xyz_rpy - dataset_statistics['action']['mean'][:-1]) / (dataset_statistics['action']['std'][:-1] + 1e-8)
        result_dict['action'] = tf.concat([xyz_rpy, tf.expand_dims(grip_action, axis=-1)], axis=-1)
        pad_grip_action = tf.pad(grip_action, [[1, 0]], constant_values=1)
        result_dict['delta_grip_act'] = pad_grip_action[1:] - pad_grip_action[:-1]
        # result_dict['delta_grip_act'] = result_dict['action'][1:, -1] - result_dict['action'][:-1, -1]
        # result_dict['delta_grip_act'] = tf.concat([tf.zeros((1,), dtype=tf.float32), result_dict['delta_grip_act']], axis=0)
        result_dict['language_instruction'] = steps['language_instruction']
        result_dict = chunk_window(result_dict, window_size, future_step_scale*dataset_freq)
        result_dict['tfds_id'] = tf.repeat(demo['tfds_id'], traj_len)
        return result_dict
    
    def non_zero_length(demo):
        return tf.shape(demo['steps']['action'])[0] > 1
    
    original_length = len(dataset)
    dataset = dataset.map(demo_std_transform, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(non_zero_length)
    
    dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, dataset_statistics, original_length

metrics_registry = {}
def register_metric(name):
    def decorator(func):
        metrics_registry[name] = func
        return func
    return decorator

def get_action(demo, chunked):
    if chunked:
        action = demo['action'][:, -1]
    else:
        action = demo['action']
    return action

def get_delta_grip_act(demo, chunked):
    if chunked:
        delta_grip_act = demo['delta_grip_act'][:, -1]
    else:
        delta_grip_act = demo['delta_grip_act']
    return delta_grip_act

@register_metric("small_action")
def small_action(demo, dataset_statistics, chunked=True):
    proprio = demo['proprio']
    abs_proprio_change = tf.abs(proprio[1:] - proprio[:-1])
    mean_proprio_change = tf.reduce_mean(abs_proprio_change, axis=0)
    is_small_proprio_change = tf.reduce_all(abs_proprio_change < 0.1 * mean_proprio_change, axis=-1)
    is_small_proprio_change = tf.concat([is_small_proprio_change, [False]], axis=0)
    
    action = get_action(demo, chunked)
    grip_action = action[:, -1]
    xyz_rpy = action[:, :-1]
    mean_action_scale = tf.reduce_mean(tf.abs(xyz_rpy), axis=0)
    is_small_action = tf.reduce_all(tf.abs(xyz_rpy) <= 0.1 * mean_action_scale, axis=-1)
    is_small_action_and_not_moving = is_small_action & (grip_action == 1) & is_small_proprio_change
    # small_action_frames = tf.nest.map_structure(lambda x: tf.boolean_mask(x, is_small_action_and_not_moving), demo)
    return is_small_action_and_not_moving

@register_metric("fail_grasp")
def fail_grasp(demo, dataset_statistics, chunked=True, pre_frames=10, word_list=["put","fold", "cloth", "towel", "pan", "pot"]):
    if 'language_instruction' in demo:
        language_inst = demo['language_instruction'][0]
    else:
        language_inst = demo['task']['language_instruction'][0]
        
    pattern = r".*\b(" + r"|".join(word_list) + r")\b.*"
    exist = tf.strings.regex_full_match(language_inst, pattern)
    
    if tf.shape(demo['proprio'])[-1] != 7 and tf.shape(demo['proprio'])[-1] != 8 or exist:
        return tf.zeros((tf.shape(demo['action'])[0],), dtype=tf.bool)
    fully_closed_grip = demo['proprio'][:, -1] < dataset_statistics['proprio']['p01'][-1]
    action = get_action(demo, chunked)
    
    moving_up = action[: ,2] > tfp.stats.percentile(action[:, 2], 80.0)
    fail_grasp = fully_closed_grip & moving_up
    for i in range(pre_frames):
        fail_grasp = fail_grasp | tf.roll(fail_grasp, -1, axis=0)
    # fail_grasp_frames = tf.nest.map_structure(lambda x: tf.boolean_mask(x, fail_grasp), demo)
    return fail_grasp

@register_metric("jittering")
def jittering(demo, dataset_statistics, chunked=True):
    action = get_action(demo, chunked)
    action = action - tf.reduce_mean(action, axis=0)
    jittering = tf.reduce_sum(action[1:, :-1] * action[:-1, :-1], axis=-1)/(tf.linalg.norm(action[1:, :-1], axis=-1)+1e-9)/(tf.linalg.norm(action[:-1, :-1], axis=-1)+1e-9) < -0.2
    jittering = tf.concat([tf.zeros((1,), dtype=bool), jittering], axis=0)
    jittering = jittering & tf.roll(jittering, 1, axis=0)
    close_direction = tf.reduce_sum(action[2:, :-1] * action[:-2, :-1], axis=-1)/(tf.linalg.norm(action[2:, :-1], axis=-1)+1e-9)/(tf.linalg.norm(action[:-2, :-1], axis=-1)+1e-9) > 0.7
    close_direction = tf.concat([tf.zeros((2,), dtype=bool), close_direction], axis=0)
    jittering = jittering | tf.roll(jittering, 1, axis=0)
    
    # jittering_frames = tf.nest.map_structure(lambda x: tf.boolean_mask(x, jittering), demo)
    return jittering  

@register_metric("multiple_gripper_close")
def delta_gripper_act(demo, dataset_statistics, chunked=True, get_pre_release_mask=False, multiple=True):
    delta_grip_act = get_delta_grip_act(demo, chunked)
    grasp = delta_grip_act < -0.5
    release = delta_grip_act > 0.5
    pre_release = tf.roll(release, -1, axis=0)
    num_grasp = tf.reduce_sum(tf.cast(grasp, tf.int32))
    num_release = tf.reduce_sum(tf.cast(release, tf.int32))
    assert num_grasp == num_release or num_grasp == num_release + 1
    
    grasp_idx = tf.where(grasp)
    pre_release_idx = tf.where(pre_release)
    
    if num_grasp == num_release + 1:
        pre_release_idx = tf.concat([pre_release_idx, [[-1]]], axis=0)
    
    def change_future_image(future_image, image, num_grasp, grasp_idx, pre_release_idx):
        future_image = future_image.numpy()
        for i in range(num_grasp):
            future_image[grasp_idx[i][0]] = image[pre_release_idx[i][0], -1].numpy()
        future_image = tf.convert_to_tensor(future_image)
        return future_image
    
    demo['future_image'] = tf.py_function(change_future_image, [demo['future_image'], demo['image'], num_grasp, grasp_idx, pre_release_idx], Tout=tf.uint8)
    
    if num_grasp < 2 and multiple:
        mask = tf.zeros((tf.shape(demo['action'])[0],), dtype=tf.bool)
        pre_release_mask = tf.zeros((tf.shape(demo['action'])[0],), dtype=tf.bool)
    else:    
        mask = grasp
        pre_release_mask = pre_release
    if get_pre_release_mask:
        return mask, pre_release_mask
    else:
        return mask

def save_suboptimal_frames(sub_op_frames, hdf5_file):
    sub_len = tf.shape(sub_op_frames['action'])[0]
    # if end_idx > hdf5_file["image"].shape[0]:
    file_len = hdf5_file["image"].shape[0]
    resized_len = int(file_len+sub_len.numpy().item())
    for key in hdf5_file:
        hdf5_file[key].resize(resized_len, axis=0)
        hdf5_file[key][-sub_len:] = sub_op_frames[key]

def visualize_suboptimal_frames(demo, sub_mask, visualize_dir, metric):
    # sub_len = tf.shape(sub_op_frames['action'])[0]
    image = demo['image'].numpy()
    sub_mask = sub_mask
    language_instruction = demo['language_instruction'][0].numpy().decode()
    image[sub_mask] = apply_filter(image[sub_mask], (255,0,0))
    
    save_dir = os.path.join(visualize_dir, metric)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    video_name = os.path.join(save_dir, f"{language_instruction}_{metric}_{time_stamp}.mp4")
    imageio.mimsave(video_name, image[:, -1], fps=5)

def collect_suboptimal_classifier_dataset(name, data_dir, metrics, save_dir:str=None, hdf5_file:h5py.File=None, visualize:int=0, visualize_dir:str=None, collect_rand:int=0, window_size:int=2, future_step_scale:int=1, split:str='val'):
    dataset, dataset_statistics, length = get_datasets(name, data_dir, window_size, future_step_scale, split)
    visualize_remain = {metric: visualize for metric in metrics}
    sub_optimal_sum = {metric: 0 for metric in metrics}
    total_sum = 0
    traj_meta = {"summary":{}, "id_mapping": {}, "subop_metric_number":{}}
    info_arrays = []
    
    if hdf5_file is not None:
        if collect_rand > 0:
            metrics.append("rand")
        for i,metric in enumerate(metrics):
            traj_meta['subop_metric_number'][metric] = i
            hdf5_file.create_group(metric)
            hdf5_file[metric].create_dataset("image", (0, window_size, 256, 256, 3), maxshape=(None, window_size, 256, 256, 3), dtype='uint8')
            hdf5_file[metric].create_dataset("action", (0, window_size, 7), maxshape=(None, window_size, 7), dtype='float32')
            hdf5_file[metric].create_dataset("future_action", (0, window_size, 7), maxshape=(None, window_size, 7), dtype='float32')
            hdf5_file[metric].create_dataset("language_instruction", (0,), maxshape=(None,), dtype='S100')
            hdf5_file[metric].create_dataset("norm_delta_proprio", (0, window_size), maxshape=(None, window_size), dtype='float32')
            hdf5_file[metric].create_dataset("delta_grip_act", (0, window_size), maxshape=(None, window_size), dtype='float32')
            hdf5_file[metric].create_dataset("tfds_id", (0,), maxshape=(None,), dtype='S100')
            hdf5_file[metric].create_dataset("future_image", (0, 256, 256, 3), maxshape=(None, 256, 256, 3), dtype='uint8')
        if collect_rand > 0:
            metrics.remove("rand")
    
    for i, demo in tqdm(enumerate(dataset), total=length):
        decode_tfds_id = demo['tfds_id'][0].numpy().decode()
        traj_meta['id_mapping'][decode_tfds_id] = {}
        traj_len = tf.shape(demo['action'])[0].numpy().item()
        info_array = np.zeros((traj_len, len(metrics)), dtype=bool)
        
        traj_meta['id_mapping'][decode_tfds_id]["all"] = (total_sum, total_sum + traj_len)
        total_sum += traj_len
        total_sub_mask = tf.zeros((traj_len,), dtype=tf.bool)

        for j,metric in enumerate(metrics):
            metric_fn = metrics_registry[metric]
            sub_mask = metric_fn(demo, dataset_statistics)
            info_array[:, j] = sub_mask.numpy()
            sub_len = tf.reduce_sum(tf.cast(sub_mask, tf.int32))
            if sub_len > 0:
                sub_len = sub_len.numpy().item()
                traj_meta['id_mapping'][decode_tfds_id][metric] = (sub_optimal_sum[metric], sub_optimal_sum[metric] + sub_len)
                sub_op_frames = tf.nest.map_structure(lambda x: tf.boolean_mask(x, sub_mask), demo)
                sub_optimal_sum[metric] += sub_len
                if hdf5_file is not None:
                    save_suboptimal_frames(sub_op_frames, hdf5_file[metric])
                if visualize_remain[metric] > 0:
                    visualize_suboptimal_frames(demo, sub_mask, visualize_dir, metric)
                    visualize_remain[metric] -= 1
                total_sub_mask = total_sub_mask | sub_mask
            else:
                traj_meta['id_mapping'][decode_tfds_id][metric] = None
                
        info_arrays.append(info_array)
                
        if collect_rand > 0 and hdf5_file is not None:
            rand_num = random.randint(0, 1)
            if rand_num > 0:
                not_sub_mask = ~total_sub_mask
                not_sub_idx = tf.where(not_sub_mask).numpy().flatten()
                rand_idx = random.choices(not_sub_idx, k=1)
                rand_frames = tf.nest.map_structure(lambda x: tf.gather(x, rand_idx), demo)
                save_suboptimal_frames(rand_frames, hdf5_file["rand"])   
                collect_rand -= 1     
            
    pprint(sub_optimal_sum)
    print(f"Total number of frames: {total_sum}")
    traj_meta['summary'] = {**sub_optimal_sum, "total": total_sum}
    if save_dir is not None:
        sub_dir = os.path.join(save_dir, name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        with open(os.path.join(sub_dir, "traj_meta.pkl"), 'wb') as f:
            pickle.dump(traj_meta, f)
        concat_info_array = np.concatenate(info_arrays, axis=0)
        np.save(os.path.join(sub_dir, "info_arrays.npy"), np.array(concat_info_array))
            
    
def main(data_dir:str, name:list[str]=None, save_dir:str=None, metrics:list[str]=["small_action", "fail_grasp", "jittering"], visualize:int=0, visualize_dir:str=None, collect_rand:int=0, window_size:int=2, future_step_scale:int=2, split:str='val', skip_datasets:list[str]=[""]):
    datasets = os.listdir(data_dir)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        hdf5_file = h5py.File(os.path.join(save_dir, "suboptimal_frames.hdf5"), 'w')
        
    for i, dataset in enumerate(datasets):
        if dataset in skip_datasets:
            print(f"Skipping {dataset}")
            continue
        if name is not None and dataset not in name:
            continue
        if save_dir is not None:
            if dataset in hdf5_file:
                continue
            hdf5_file.create_group(dataset)
            
        print(f"Processing {dataset}, {i+1}/{len(datasets)}")
        if dataset not in OXE_DATASET_CONFIGS:
            continue
        if OXE_DATASET_CONFIGS[dataset]['image_obs_keys']['primary'] == None:
            continue
        collect_suboptimal_classifier_dataset(dataset, data_dir, metrics, save_dir, None if save_dir is None else hdf5_file[dataset], visualize=visualize, visualize_dir=visualize_dir, collect_rand=collect_rand, window_size=window_size, future_step_scale=future_step_scale, split=split)
    
    if save_dir is not None:
        hdf5_file.close()
    
if __name__ == "__main__":
    tyro.cli(main)