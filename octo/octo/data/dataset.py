from functools import partial
import json
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from octo.data import obs_transforms, traj_transforms
from octo.data.utils import goal_relabeling, task_augmentation
from octo.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    sample_match_keys_uniform,
    tree_map,
)
from octo.utils.spec import ModuleSpec
import pickle
import os
from octo.data.oxe.oxe_dataset_configs import OXE_DATASET_CONTROL_FREQUENCY
from copy import deepcopy
import jax
from curation.suboptimal_classifier.dataset.collect_sub import jittering, fail_grasp, small_action

def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    action_horizon: int = 1,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    max_action_dim: Optional[int] = None,
    max_proprio_dim: Optional[int] = None,
    post_chunk_transforms: Sequence[ModuleSpec] = (),
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of
    "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that happen in this
    function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The window size to chunk both observations and actions into.
        action_horizon (int, optional): The size of the action chunk (present and future actions) to include in
            the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        max_action_dim (int, optional): If provided, datasets with an action dimension less than this will be
            padded to this dimension.
        max_proprio_dim (int, optional): If provided, datasets with a proprio dimension less than this will be
            padded to this dimension.
        post_chunk_transforms (Sequence[ModuleSpec]): ModuleSpecs of trajectory transforms applied after
            chunking.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError(
                "skip_unlabeled=True but dataset does not have language labels."
            )
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )
    # marks which entires of the observation and task dicts are padding
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # optionally pads actions and proprio to a consistent number of dimensions
    dataset = dataset.traj_map(
        partial(
            traj_transforms.pad_actions_and_proprio,
            max_action_dim=max_action_dim,
            max_proprio_dim=max_proprio_dim,
        ),
        num_parallel_calls,
    )

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            action_horizon=action_horizon,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    for transform_spec in post_chunk_transforms:
        transform_fn = ModuleSpec.instantiate(transform_spec)
        dataset = dataset.traj_map(
            transform_fn,
            num_parallel_calls,
        )

    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    image_dropout_prob: float = 0.0,
    image_dropout_keep_key: Optional[str] = None,
    do_transform: bool = True,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        image_dropout_prob (float): Probability of dropping out images, applied to each image key
            independently. At least one image will always be present.
        image_dropout_keep_key (str, optional): Optionally provide a key to always keep during image dropout
            for example for image observations that are essential for action prediction.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        # task is not chunked -- apply fn directly
        frame["task"] = fn(frame["task"])
        # observation is chunked -- apply fn along first axis
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame
    
    # decode + resize images (and depth images)
    def dummy_decode_and_resize(obs, resize_size, depth_resize_size):
        image_names = {key[6:] for key in obs if key.startswith("image_")}
        for name in image_names:
            if resize_size.get(name, None) is not None:
                obs[f"image_{name}"] = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
                
        return obs
    
    decode_and_resize_fn = partial(
        apply_obs_transform,
        partial(
            obs_transforms.decode_and_resize,
            resize_size=resize_size,
            depth_resize_size=depth_resize_size,
        )
    )
    if do_transform:
        dataset = dataset.frame_map(
            decode_and_resize_fn,
            num_parallel_calls,
        )
    if train:
        # augment all images with the same seed, skipping padding images
        def aug_and_dropout(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            dropout_fn = partial(
                obs_transforms.image_dropout,
                seed=seed,
                dropout_prob=image_dropout_prob,
                always_keep_key=image_dropout_keep_key,
            )
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            frame = apply_obs_transform(dropout_fn, frame)
            frame = apply_obs_transform(aug_fn, frame)                
            return frame
        if do_transform:
            dataset = dataset.frame_map(aug_and_dropout, num_parallel_calls)

    return dataset

def remove_id(traj):
    traj = {k: v for k, v in traj.items() if k != 'tfds_id'}
    return traj

def remove_proprio(traj):
    traj['observation'] = {k: v for k, v in traj['observation'].items() if k != 'proprio'}
    traj = {k: v for k, v in traj.items() if k != 'proprio'}
    return traj

def apply_dedup_curation_mask(
    dataset: dl.DLataset,
    dataset_name: str,
    curation_df_path: str,
    curation_eps: list[str],
    curation_prob: list[float],
    chunk_time: float,
    random_curate: bool = False,
    curation_rebalance: bool = False,
    density_sample: bool = False,
    seed: int = 0,
    **kwargs,
) -> dl.DLataset:
    chunk_length = int(chunk_time * OXE_DATASET_CONTROL_FREQUENCY[dataset_name])
    def curate(tfds_id, prob, eps, traj_len):
        tfds_id = tfds_id[0].numpy().decode()
        eps = eps.numpy()
        dedup_mask = np.zeros(traj_len, dtype=bool)
        df = curation_df_dict[f'eps={eps:g}']
        for start in range(0, traj_len, chunk_length):
            end = min(start + chunk_length, traj_len)
            chunk_id = f'{tfds_id}-{start}-{end}'
            curated = df.get(chunk_id, False)
            if curated:
                dedup_mask[start:end] = np.random.rand() < prob
        return dedup_mask
    
    def curate_wrapper(traj, prob, eps):
        traj_len = tf.shape(traj['action'])[0]
        dedup_curate_mask = tf.py_function(curate, [traj['tfds_id'], prob, eps, traj_len], tf.bool)
        traj['dedup_curate_mask'] = dedup_curate_mask
        return traj
    
    with open(curation_df_path, 'rb') as f:
        curation_df = pickle.load(f)
    
    current_dataset_curation_df = curation_df[np.char.count(curation_df.index.to_numpy().astype(str), dataset_name)>0]
    
    curated_dataset =  dataset
    prev_curation_df = None
    curation_eps = [float(x) for x in curation_eps.split(',')]
    curation_prob = [float(x) for x in curation_prob.split(',')]
    curation_prob = curation_prob[:len(curation_eps)]
    assert len(curation_eps) >= len(curation_prob), "curation_eps and curation_prob must have the same length"
    curation_df_dict = {}
    
    for eps, prob in zip(curation_eps, curation_prob):
        if not density_sample:
            if not curation_rebalance:
                thres = np.percentile(np.nan_to_num(current_dataset_curation_df['max_sim']), 100*eps)
            else:
                thres = np.percentile(np.nan_to_num(curation_df['max_sim']), 100*eps)
            current_curation_df = current_dataset_curation_df['max_sim'] > thres
        else:
            if not curation_rebalance:
                density_array = current_dataset_curation_df['density'].to_numpy()
            else:
                density_array = curation_df['density'].to_numpy()
                
            # inverse density sampling
            weight = 1 / (density_array + 1e-10)
            sample_prob = weight / np.sum(weight)
            np_rng = np.random.default_rng(seed)    
            sample_indices = np_rng.choice(np.arange(len(density_array)), size=int(len(density_array) * eps), replace=False, p=sample_prob)
            current_dataset_curation_df['sampled'] = True
            current_dataset_curation_df.loc[current_dataset_curation_df.index[sample_indices], 'sampled'] = False
            current_curation_df = current_dataset_curation_df['sampled']
            
        split_chunk_ids = np.char.split(current_curation_df.index.to_numpy().astype(str), '-')
        curate_sizes = [int(split_chunk_id[-1])-int(split_chunk_id[-2]) for i,split_chunk_id in enumerate(split_chunk_ids) if current_curation_df.iloc[i]]
        if prev_curation_df is not None:
            current_curation_df = current_curation_df & ~prev_curation_df
            prev_curation_df = current_curation_df | prev_curation_df
        else:
            current_curation_df = current_curation_df
            prev_curation_df = current_curation_df
        
        if random_curate:
            curate_np = current_curation_df.to_numpy()
            np_rng = np.random.default_rng(seed)
            np_rng.shuffle(curate_np)
            current_curation_df.loc[:] = curate_np
            
        curation_df_dict[f'eps={eps:g}'] = deepcopy(current_curation_df)
        curated_dataset = curated_dataset.map(partial(curate_wrapper, prob=prob, eps=eps))
        prev_curation_df = current_curation_df
        
    # def cnt_dedup_fn(x, traj): 
    #     return x + tf.reduce_sum(tf.cast(~traj['dedup_curate_mask'], tf.int32))
    # def cnt_fn(x, traj): 
    #     return x + tf.shape(traj['action'])[0]
    # print(f"Curating dataset with eps", curation_eps , "prob:", curation_prob,"start counting")
    # original_size = dataset.reduce(0, cnt_fn)
    # curated_size = curated_dataset.reduce(0, cnt_dedup_fn)
    # print(f"Original dataset size: {original_size}, Curated dataset size: {curated_size}")
    return curated_dataset, sum(curate_sizes)

def apply_pivot_reweighting(
    dataset: dl.DLataset,
    name: str,
    pivot_df_path: str,
    **kwargs,
) -> dl.DLataset:
    
    def get_pivot_tensor(tfds_id, action):
        shape = tf.shape(action)
        pivot = pivot_df.get(tfds_id[0].numpy(), None)
        if pivot is not None and pivot.shape[0] == shape[0]:
            pivot_tensor = tf.convert_to_tensor(pivot, dtype=tf.bool)
        else:
            pivot_tensor = tf.zeros((shape[0],), dtype=tf.bool)
        return pivot_tensor
    
    def get_pivot(traj):
        pivot_tensor = tf.py_function(get_pivot_tensor, [traj['tfds_id'], traj['action']], tf.bool)
        traj['action_pivot'] = pivot_tensor
        return traj
    
    # breakpoint()
    with open(os.path.join(pivot_df_path, name, "features.pkl"), 'rb') as f:
        pivot_df = pickle.load(f)
        pivot_df = pivot_df.set_index('id')
        pivot_df = pivot_df['final']
        
    add_pivot_dataset = dataset.map(get_pivot)
    
    return add_pivot_dataset

def apply_freq_unify(
    dataset: dl.DLataset,
    name: str,
    base_control_freq: int,
    **kwargs,
):
    freq = OXE_DATASET_CONTROL_FREQUENCY[name]
    if freq == None:
        return dataset
    subsample_ratio = freq // base_control_freq
    if subsample_ratio <= 1:
        return dataset
    
    def subsample(traj):
        # sum the action over the subsample_ratio
        traj_len = tf.shape(traj['action'])[0]
        sum_action = tf.zeros_like(traj['action'])
        for i in range(subsample_ratio):
            roll = tf.roll(tf.pad(traj['action'], [[0, i], [0, 0]]), shift=-i, axis=0)
            sum_action += roll[:traj_len]
        # binarize grip action
        grip_action = tf.cast(sum_action[:, -1] >= 0.5, tf.float32)
        sum_action = tf.concat([sum_action[:, :-1], grip_action[:, None]], axis=-1)
        
        sub_traj = tf.nest.map_structure(lambda x: x[::subsample_ratio], traj)
        sub_traj['action'] = sum_action[::subsample_ratio]
        return sub_traj
    
    subsample_dataset = dataset.map(subsample)
    return subsample_dataset
  
def apply_small_action_filter(
    dataset: dl.DLataset,
    thres: float
):  
    def filter_small_action(traj):
        action = traj['action']
        gripper_action = action[:, -1]
        other_action = action[:, :-1]
        action_norm = tf.norm(other_action, axis=-1)
        small_action = tf.logical_and(action_norm < thres, gripper_action == 1.0)
        reasonable_action = tf.logical_not(small_action)
        traj_left = tf.nest.map_structure(lambda x: tf.boolean_mask(x, reasonable_action), traj)
        return traj_left
    
    filtered_dataset = dataset.map(filter_small_action)
    return filtered_dataset

def add_hardcode_subop_curation_tag(dataset: dl.DLataset, dataset_statistics):
    def filter_suboptimal(traj):
        suboptimal_mask = tf.zeros(tf.shape(traj['action'])[0], dtype=tf.bool)
        traj["proprio"] = traj["observation"]["proprio"]
        jittering_mask = jittering(traj, dataset_statistics, chunked=False)
        suboptimal_mask = tf.logical_or(suboptimal_mask, jittering_mask)
        fail_grasp_mask = fail_grasp(traj, dataset_statistics, chunked=False)
        suboptimal_mask = tf.logical_or(suboptimal_mask, fail_grasp_mask)
        small_action_mask = small_action(traj, dataset_statistics, chunked=False)
        suboptimal_mask = tf.logical_or(suboptimal_mask, small_action_mask)
        
        traj['hardcode_suboptimal_mask'] = suboptimal_mask
        return traj
    
    filtered_dataset = dataset.map(filter_suboptimal)
    
    return filtered_dataset

def get_subop_file(dataset_name, curation_args):
    score_path = curation_args.get('subop_score_path', None)
    dataset_subop_dir_path = os.path.join(score_path, dataset_name)
    score_file = os.path.join(dataset_subop_dir_path, "scores.npy")
    traj_meta = os.path.join(dataset_subop_dir_path, "traj_meta.pkl")
    if not os.path.exists(score_file) or not os.path.exists(traj_meta):
        logging.info(f"\nSuboptimal mask file {score_file} or traj_meta file {traj_meta} does not exist, skipping suboptimal mask curation!!!\n")
        subop_score, traj_meta = None, None
    else:
        subop_score = np.load(score_file, allow_pickle=True)
        with open(traj_meta, 'rb') as f:
            traj_meta = pickle.load(f)['traj_meta']
    return subop_score, traj_meta

def window_average(scores, traj_meta, curation_args, dataset_name):
    average_time = curation_args.get('average_time', 0.0)
    mix_level = curation_args.get('subop_mix_level', 0.0)
    
    average_window = int(OXE_DATASET_CONTROL_FREQUENCY[dataset_name] * average_time)
    average_window = max(average_window, 1)
    assert len(scores.shape) == 1, f"Scores shape {scores.shape} is not 1D"
    pad_width = (average_window//2, average_window//2+1)
    
    from concurrent.futures import ThreadPoolExecutor
    def smooth_score(tfds_id):
        start, end = traj_meta[tfds_id]['start'], traj_meta[tfds_id]['end']
        cur_score = scores[start:end]
        if curation_args.get('traj_level_subop', False):
            scores[start:end] = np.ones_like(cur_score) * np.mean(cur_score)
        elif average_window > 1:
            pad_score = np.pad(cur_score, pad_width, mode='edge')
            smoothed_score = np.convolve(pad_score, np.ones(average_window)/average_window, mode='same')
            scores[start:end] = smoothed_score[pad_width[0]:-pad_width[1]]
        else: # no smoothing or traj level averaging
            pass
        if mix_level > 0.0:
            mean_score = np.mean(cur_score)
            scores[start:end] = mix_level * mean_score + (1-mix_level) * scores[start:end]

    jobs = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for tfds_id in traj_meta:
            jobs.append(executor.submit(smooth_score, tfds_id))
        
    for job in jobs:
        job.result()
        
    return scores
        
        
def add_subop_score(dataset, dataset_name, curation_args):
    subop_mask_path = curation_args.get('subop_score_path', None)
    dataset_subop_dir_path = os.path.join(subop_mask_path, dataset_name)
    score_file = os.path.join(dataset_subop_dir_path, "scores.npy")
    traj_meta = os.path.join(dataset_subop_dir_path, "traj_meta.pkl")
    subop_score_type = curation_args.get('subop_score_type', 'suboptimal_score')
    delta_time = curation_args.get('delta_time', 2.0)
    delta_step = int(OXE_DATASET_CONTROL_FREQUENCY[dataset_name] * delta_time)
    traj_level_subop = curation_args.get('traj_level_subop', False)
    thres = curation_args.get('subop_score_thres', 1.0)
    start_thres = curation_args.get('subop_score_start_thres', 1.0)
    per_dataset_subop_thres = curation_args.get('per_dataset_subop_thres', False)
    
    if not os.path.exists(score_file) or not os.path.exists(traj_meta):
        logging.info(f"\nSuboptimal score file {score_file} or traj_meta file {traj_meta} does not exist, skipping suboptimal score curation!!!\n")
        scores, traj_meta, thres = None, None, 1.0
    else:
        scores = np.load(score_file, allow_pickle=True)
        with open(traj_meta, 'rb') as f:
            traj_meta = pickle.load(f)['traj_meta']
        if not traj_level_subop:
            scores = scores.flatten()
            scores = window_average(scores, traj_meta, curation_args, dataset_name)
        else:
            for tfds_id in traj_meta:
                start, end = traj_meta[tfds_id]['start'], traj_meta[tfds_id]['end']
                scores[start:end] = np.ones_like(scores[start:end]) * np.mean(scores[start:end])
        if per_dataset_subop_thres:
            thres = np.percentile(scores, curation_args.get('subop_score_keep_percentile', 1.0)*100)
            start_thres = np.percentile(scores, curation_args.get('subop_score_start_curate_percentile', 1.0)*100)
    
    def get_subop_score(tfds_id):
        traj_len = tf.shape(tfds_id)[0]
        tfds_id = tfds_id[0].numpy()
        if scores is not None:
            meta = traj_meta.get(tfds_id, None)
            if meta is not None:
                start, end = meta['start'], meta['end']
                subop_score = scores[start:end]
                if subop_score_type == 'suboptimal_score':
                    subop_score = subop_score
                elif subop_score_type == 'progress':
                    subop_score = np.concatenate([subop_score, [subop_score[-1]]*delta_step])
                    subop_score = (subop_score[delta_step:] - subop_score[:-delta_step]) < 0
            else:
                subop_score = tf.zeros((traj_len,), dtype=tf.float32)
        else:
            subop_score = tf.zeros((traj_len,), dtype=tf.float32)
            
        assert subop_score.shape[0] == traj_len, f"subop_score shape {subop_score.shape} does not match action shape {traj_len}"
        subop_score = tf.convert_to_tensor(subop_score, dtype=tf.float32)
        subop_score = tf.reshape(subop_score, (traj_len,))
        return subop_score
    
    def get_subop_score_wrapper(traj):
        tfds_id = traj['tfds_id']
        subop_score = tf.py_function(get_subop_score, [tfds_id], tf.float32)
        traj_len = tf.shape(traj['action'])[0]
        traj['suboptimal_score'] = subop_score
        traj['sampling_prob'] = tf.random.uniform([traj_len], minval=0, maxval=1, dtype=tf.float32)
        if curation_args.get('act_score_cond', False):
            traj['act_score'] = subop_score
            traj['act_score'] = tf.reshape(traj['act_score'], (traj_len, ))
        traj['subop_score_mask'] = (subop_score > thres) & (subop_score < start_thres)
        return traj
    dataset_with_score = dataset.map(get_subop_score_wrapper)
    return dataset_with_score, scores, thres, start_thres

def get_subop_score_thres(dataset_kwargs_list, curation_args):
    subop_score_list = []
    for dataset_kwargs in dataset_kwargs_list:
        dataset_name = dataset_kwargs['name']
        subop_score, traj_meta = get_subop_file(dataset_name, curation_args)
        if subop_score is not None:
            subop_score = subop_score.flatten()
            subop_score = window_average(subop_score, traj_meta, curation_args, dataset_name)
            subop_score_list.append(subop_score)
    if len(subop_score_list) == 0:
        return 1.0
    subop_score = np.concatenate(subop_score_list)
    keep_percentile = curation_args.get('subop_score_keep_percentile', 1.0)
    start_percentile = curation_args.get('subop_score_start_curate_percentile', 1.0)
    thres = np.percentile(subop_score, keep_percentile*100)
    start_thres = np.percentile(subop_score, start_percentile*100)
    logging.info(f"\nSuboptimal score threshold is {thres}\n")
    return thres, start_thres

def apply_frame_level_curation(dataset: dl.DLataset, curation_args: dict):
    logging.info(f"\nApplying frame level curation\n")
    thres = curation_args.get('subop_score_thres', 1.0)
    hardcode_curate = curation_args.get('hardcode_subop_curation', False) 
    subop_score_path = curation_args.get('subop_score_path', False)
    subop_sampling = curation_args.get('subop_sampling', False)
    delta_gripping_ratio = curation_args.get("delta_gripping_act_ratio", 0.0)
    use_dedup = curation_args.get('save_folder', None) is not None
    act_score_cond = curation_args.get('act_score_cond', False)
    delta_gripping = delta_gripping_ratio > 0.055
    
    def filter_fn(frame):
        to_curate = False
        if hardcode_curate:
            to_curate = tf.logical_or(to_curate, frame['hardcode_suboptimal_mask'])
        if subop_score_path and not act_score_cond:
            if subop_sampling:
                to_curate = tf.logical_or(to_curate, frame['sampling_prob'] < (frame['suboptimal_score'] - thres) / (1 - thres) )
            else:
                to_curate = tf.logical_or(to_curate, frame['subop_score_mask'])
        if 'dedup_curate_mask' in frame:  # deduplication
            to_curate = tf.logical_or(to_curate, frame['dedup_curate_mask'])
        return not to_curate
        
    if hardcode_curate or subop_score_path or use_dedup:
        dataset = dataset.filter(filter_fn)
        
    def delta_grip_resampling(frame):    
        is_delta_gripping = frame['observation']['delta_gripping'][-1] !=0
        prob = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
        return is_delta_gripping or prob < rescale_ratio
    
    if delta_gripping:
        rescale_ratio = 0.055/delta_gripping_ratio*(1-delta_gripping_ratio)/(1-0.055) # 0.055 is the rough ratio of delta gripping in the raw dataset
        dataset = dataset.filter(delta_grip_resampling)
    
    return dataset        
    
def calculate_norm_delta_proprio(dataset: dl.DLataset, dataset_statistics: dict):
    def get_norm(traj):
        traj['observation']['delta_proprio'] = traj['observation']['delta_proprio'] / (dataset_statistics['proprio']['std']+1e-8)
        traj['observation']['delta_proprio'] = tf.norm(traj['observation']['delta_proprio'], axis=-1)
        return traj

    dataset_with_delta_proprio = dataset.map(get_norm)
    return dataset_with_delta_proprio

def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[ModuleSpec] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    proprio_obs_key: Optional[str] = None,
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    force_recompute_dataset_statistics: bool = False,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    filter_functions: Sequence[ModuleSpec] = (),
    skip_norm: bool = False,
    ignore_errors: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    curation_args: dict = {},
) -> Tuple[dl.DLataset, dict]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will
    contain the key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset,
            since one file usually contains many trajectories!).
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        proprio_obs_key (str, optional): If provided, the "obs" dict will contain the key "proprio", extracted from
            `traj["observation"][proprio_obs_key]`.
        language_key (str, optional): If provided, the "task" dict will contain the key
            "language_instruction", extracted from `traj[language_key]`. If language_key fnmatches multiple
            keys, we sample one uniformly.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. May also provide "num_transitions" and "num_trajectories" keys for downstream usage
            (e.g., for `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        force_recompute_dataset_statistics (bool, optional): If True and `dataset_statistics` is None, will
            recompute the dataset statistics regardless of whether they are already cached.
        action_normalization_mask (Sequence[bool], optional): If provided, only normalizes action dimensions
            where the corresponding mask is True. For example, you might not want to normalize the gripper
            action dimension if it's always exactly 0 or 1. By default, all action dimensions are normalized.
        filter_functions (Sequence[ModuleSpec]): ModuleSpecs for filtering functions applied to the
            raw dataset.
        skip_norm (bool): If true, skips normalization of actions and proprio. Default: False.
        ignore_errors (bool): If true, skips erroneous dataset elements via dataset.ignore_errors(). Default: False.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    
    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = ModuleSpec.instantiate(standardize_fn)(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if proprio_obs_key is not None:
            new_obs["proprio"] = tf.cast(old_obs[proprio_obs_key], tf.float32)

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict, or samples uniformly if `language_key` fnmatches multiple keys
        task = {}
        if language_key is not None:
            task["language_instruction"] = sample_match_keys_uniform(traj, language_key)
            if task["language_instruction"].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {task['language_instruction'].dtype}, "
                    "but it must be tf.string."
                )
                
        restruct_traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
            "is_last": traj["is_last"],
            "is_first": traj["is_first"],
            "tfds_id": traj["tfds_id"],
        }
        # if curation_args.get("hardcode_subop_curation", False) and train:
        restruct_traj['observation']['proprio'] = traj['observation']['proprio']
        if curation_args.get("load_delta_proprio", False):
            # restruct_traj['observation']['proprio'] = traj['observation']['proprio']
            restruct_traj['observation']['delta_proprio'] = traj['observation']['proprio'][1:] - traj['observation']['proprio'][:-1]
            restruct_traj['observation']['delta_proprio'] = restruct_traj['observation']['delta_proprio'][:, -1:]
            restruct_traj['observation']['delta_proprio'] = tf.concat([restruct_traj['observation']['delta_proprio'], restruct_traj['observation']['delta_proprio'][-1:]], axis=0)
            
        if curation_args.get("load_delta_gripping_act", False) and train:
            restruct_traj['observation']['delta_gripping'] = restruct_traj['action'][1:, -1] - restruct_traj['action'][:-1, -1]
            restruct_traj['observation']['delta_gripping'] = tf.concat([restruct_traj['observation']['delta_gripping'], tf.zeros((1,), dtype=tf.float32)], axis=0)
        return restruct_traj

    def is_nonzero_length(traj):
        return tf.shape(traj["action"])[0] > 0

    builder = tfds.builder(name, data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)
        for filter_fcn_spec in filter_functions:
            full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
        if ignore_errors:
            full_dataset = full_dataset.ignore_errors()
        full_dataset = full_dataset.traj_map(restructure).filter(is_nonzero_length)
        if curation_args.get('curation_df_path', None) is not None and train:
            full_dataset, _ = apply_dedup_curation_mask(full_dataset, name, **curation_args)
        if curation_args.get('base_control_freq', None) is not None:
            full_dataset = apply_freq_unify(full_dataset, name, **curation_args)
        full_dataset = full_dataset.traj_map(remove_id)

        base_freq = curation_args.get('base_control_freq', None)
        if curation_args.get("load_delta_proprio", False) or curation_args.get('hardcode_subop_curation', False):
            proprio_obs_key = "proprio"
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(proprio_obs_key),
                ModuleSpec.to_string(standardize_fn)
                if standardize_fn is not None
                else "",
                *map(ModuleSpec.to_string, filter_functions),
                "base_freq"+str(base_freq) if base_freq is not None else "",
            ),
            save_dir=builder.data_dir,
            force_recompute=force_recompute_dataset_statistics,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if (
            len(action_normalization_mask)
            != dataset_statistics["action"]["mean"].shape[-1]
        ):
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # construct the dataset
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    for filter_fcn_spec in filter_functions:
        dataset = dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    if ignore_errors:
        dataset = dataset.ignore_errors()

    dataset = dataset.traj_map(restructure, num_parallel_calls).filter(
        is_nonzero_length
    )    
    original_size = dataset_statistics['num_transitions']
    if curation_args.get('curation_df_path', None) is not None and train:
        dataset, curate_size = apply_dedup_curation_mask(dataset, name, **curation_args)
        logging.info(f"Deduplication curate transitions:{curate_size}. Remaining transitions:{original_size-curate_size}. Keeping ratio:{(original_size-curate_size)/original_size}")
        if curation_args.get("curation_rebalance", False):
            dataset_statistics['num_transitions'] = int((original_size - curate_size)/original_size * dataset_statistics['num_transitions'])
    if curation_args.get('pivot_df_path', None) is not None and train:
        dataset = apply_pivot_reweighting(dataset, name, **curation_args)
    if curation_args.get('curate_short_traj_len', 0) > 0 and train:
        dataset = dataset.filter(lambda x: tf.shape(x['action'])[0] > curation_args.get('curate_short_traj_len'))
    if curation_args.get('base_control_freq', None) is not None:
        dataset = apply_freq_unify(dataset, name, **curation_args)
    if curation_args.get('curate_small_action_thres', 0.0) > 0.0 and train:
        dataset = apply_small_action_filter(dataset, curation_args.get('curate_small_action_thres'))
        dataset = dataset.filter(is_nonzero_length)
    if curation_args.get('hardcode_subop_curation', False) and train:
        dataset = add_hardcode_subop_curation_tag(dataset, dataset_statistics)
    if curation_args.get('subop_score_path', False) and train:
        dataset, scores, thres, start_thres = add_subop_score(dataset, name, curation_args)
        if scores is not None:
            curate_size = np.sum((scores > thres) & (scores < start_thres))
            logging.info(f"Suboptimal score curate transitions:{curate_size}. Remaining transitions:{original_size-curate_size}. Keeping ratio:{(original_size-curate_size)/original_size}")
            if curation_args.get("curation_rebalance", False):
                dataset_statistics['num_transitions'] = int((original_size - curate_size)/original_size * dataset_statistics['num_transitions'])
    if curation_args.get('load_delta_proprio', False):
        dataset = calculate_norm_delta_proprio(dataset, dataset_statistics)
    dataset_statistics['num_transitions'] = max(dataset_statistics['num_transitions'], 0)
        
    dataset = dataset.traj_map(remove_id)
    dataset = dataset.traj_map(remove_proprio)

    if not skip_norm:
        dataset = dataset.traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=dataset_statistics,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls,
        )
    else:
        logging.warning(
            "Dataset normalization turned off -- set skip_norm=False to apply normalization."
        )

    return dataset, dataset_statistics


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    curation_args: Optional[dict] = None,
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
    curation_args: Optional[dict] = None,
    use_dali: bool = False,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overidden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overidden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # default to uniform sampling
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )
    if curation_args.get('subop_score_path', None) is not None and curation_args.get('subop_score_thres', None) is None:
        curation_args['subop_score_thres'], curation_args['subop_score_start_thres']= get_subop_score_thres(dataset_kwargs_list, curation_args)
    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = {}
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train, curation_args=curation_args)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        assert (
            dataset_kwargs["name"] not in all_dataset_statistics
        ), f"Duplicate name {dataset_kwargs['name']}"
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # allocate threads based on weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    # construct datasets
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
            curation_args=curation_args,
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        if train and curation_args.get("curate_before_sample", False):
            dataset = apply_frame_level_curation(dataset, curation_args)
        datasets.append(dataset)
        
    # dataset, _ = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train, do_transform=True)
    
    # interleave at the frame level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    )
    
    if train and not curation_args.get("curate_before_sample", False):
        dataset = apply_frame_level_curation(dataset, curation_args)
        
    dataset = dataset.shuffle(shuffle_buffer_size)
    # apply frame transforms
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train, do_transform=not use_dali)

    # sequential batch (parallel batch seems to use much more memory)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    dataset = dataset.ignore_errors(log_warning=True)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    dataset.sample_weights = sample_weights
    return dataset
