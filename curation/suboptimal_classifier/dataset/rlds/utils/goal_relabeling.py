"""
Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""

from typing import Optional

import tensorflow as tf

from curation.suboptimal_classifier.dataset.rlds.utils.data_utils import tree_merge


def uniform(traj: dict, max_goal_distance: Optional[int] = None, **kwargs) -> dict:
    """
    Relabels with a true uniform distribution over future states.
    Optionally caps goal distance.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # select a random future index for each transition i in the range [i, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len), tf.float32)
    if max_goal_distance is not None:
        high = tf.cast(
            tf.minimum(tf.range(traj_len) + max_goal_distance, traj_len), tf.float32
        )
    else:
        high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # adds keys to "task" mirroring "observation" keys (must do a tree merge to combine "pad_mask_dict" from
    # "observation" and "task" properly)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    traj["task"] = tree_merge(traj["task"], goal)

    return traj

# two frames with a random time difference from min to max
def custom_uniform(traj: dict, max_goal_time_diff: float = 2.0, min_goal_time_diff: float = -1.0, freq:int=None, **kwargs) -> dict:
    freq = 1 if freq is None else freq
    
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    # rand_dist = tf.random.uniform( [traj_len, 2], minval=min_goal_time_diff, maxval=max_goal_time_diff, dtype=tf.float32) * freq
    rand_dist_1 = tf.random.uniform( [traj_len], minval=min_goal_time_diff, maxval=min_goal_time_diff+1, dtype=tf.float32) * freq
    rand_dist_2 = tf.random.uniform( [traj_len], minval=max_goal_time_diff-1, maxval=max_goal_time_diff, dtype=tf.float32) * freq
    rand_dist = tf.stack([rand_dist_1, rand_dist_2], axis=1)
    rand_dist = tf.math.round(rand_dist)
    rand_dist = tf.cast(rand_dist, tf.int32)
    
    goal_idxs = tf.range(traj_len)[:, None] + rand_dist
    goal_idxs = tf.clip_by_value(goal_idxs, 0, traj_len-1)
    
    delta_time = tf.cast(goal_idxs - tf.range(traj_len)[:, None], tf.float32) / freq
    
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    traj['task'] = tf.nest.map_structure(lambda x:tf.repeat(x[:,None], 2, axis=1), traj["task"])
    traj["task"] = tree_merge(traj["task"], goal)
    traj["task"]["delta_time"] = delta_time
    
    return traj

# condition on goal
# observation -> goal
# task -> random
def custom_uniform_v2(traj: dict, **kwargs) -> dict:
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    rand_dist = tf.random.uniform( [traj_len, 2], minval=0, maxval=traj_len, dtype=tf.int32)
    
    delta_time = tf.cast(traj_len - rand_dist, tf.float32) / tf.cast(traj_len, tf.float32)
    
    goal = tf.nest.map_structure(lambda x: tf.gather(x, rand_dist), traj["observation"])
    traj['task'] = tf.nest.map_structure(lambda x:tf.repeat(x[:,None], 2, axis=1), traj["task"])
    traj["task"] = tree_merge(traj["task"], goal)
    traj["task"]["delta_time"] = delta_time
    
    last_idxs = tf.ones([traj_len], dtype=tf.int32) * (traj_len-1)
    traj['observation'] = tf.nest.map_structure(lambda x: tf.gather(x, last_idxs), traj['observation'])
    
    return traj


def custom_uniform_v3(traj:dict, time_bins:list[float], freq:int=None, **kwargs) -> dict:
    freq = 1 if freq is None else freq
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    
    random_bin = tf.random.uniform([traj_len, 2], minval=0, maxval=len(time_bins), dtype=tf.int32)
    bins = tf.gather(time_bins, random_bin)
    
    rand = tf.random.uniform([traj_len, 2], minval=0, maxval=1, dtype=tf.float32)
    rand_time = rand * bins[...,0] + (1-rand) * bins[...,1]
    rand_dist = tf.cast(tf.math.round(rand_time * freq), tf.int32)
    
    goal_idx = rand_dist + tf.range(traj_len)[:, None]
    goal_idx = tf.clip_by_value(goal_idx, 0, traj_len-1)
    
    delta_time = tf.cast(goal_idx - tf.range(traj_len)[:, None], tf.float32) / freq
    
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idx), traj["observation"])
    traj['task'] = tf.nest.map_structure(lambda x:tf.repeat(x[:,None], 2, axis=1), traj["task"])
    traj["task"] = tree_merge(traj["task"], goal)
    traj["task"]["delta_time"] = delta_time
    
    return traj

def custom_uniform_v4(traj:dict, time_bins:list[float], freq:int=None, **kwargs) -> dict:
    freq = 1 if freq is None else freq
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    
    random_bin = tf.random.uniform([traj_len, 1], minval=0, maxval=len(time_bins), dtype=tf.int32)
    bins = tf.gather(time_bins, random_bin)
    
    rand = tf.random.uniform([traj_len, 1], minval=0, maxval=1, dtype=tf.float32)
    rand_time = rand * bins[...,0] + (1-rand) * tf.clip_by_value(bins[...,1], 0, tf.cast(traj_len-1, tf.float32)/freq)
    rand_dist = tf.cast(tf.math.round(rand_time * freq), tf.int32)
    
    max_obs_idx = traj_len - rand_dist
    max_obs_idx = tf.clip_by_value(max_obs_idx, 0, traj_len-1)
    obs_idx = tf.random.uniform([traj_len, 1], minval=0, maxval=1, dtype=tf.float32) * tf.cast(max_obs_idx, tf.float32)
    obs_idx = tf.cast(obs_idx, tf.int32)
    obs_idx = tf.clip_by_value(obs_idx, 0, max_obs_idx-1)
    
    goal_idx = obs_idx + rand_dist
    goal_idx = tf.clip_by_value(goal_idx, 0, traj_len-1)
    
    delta_time = tf.cast(goal_idx - obs_idx, tf.float32) / freq
    
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idx), traj["observation"])
    traj['observation'] = tf.nest.map_structure(lambda x: tf.gather(x, obs_idx[:,0]), traj["observation"])
    traj['task'] = tf.nest.map_structure(lambda x:x[:,None], traj["task"])
    traj["task"] = tree_merge(traj["task"], goal)
    traj["task"]["delta_time"] = delta_time
    
    return traj

def uniform_action(traj: dict, neighbor_time:float, freq:int=None, **kwargs) -> dict:
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
    
    neighbor_dist = tf.cast(neighbor_time * freq, tf.int32)
    
    if traj_len <= 2 * neighbor_dist:
        traj['task']['delta_time'] = tf.zeros([traj_len], dtype=tf.float32)
        return traj
    
    to_random = tf.random.uniform([traj_len], minval=0, maxval=1, dtype=tf.float32) > 0.5
    rand_idx = tf.random.uniform([traj_len], minval=0, maxval=traj_len - 2 * neighbor_dist, dtype=tf.int32)
    rand_idx = tf.where(rand_idx>tf.range(traj_len)-neighbor_dist, rand_idx + 2 * neighbor_dist, rand_idx)
    
    action_idx = tf.where(to_random, rand_idx, tf.range(traj_len))
    
    traj['action'] = tf.nest.map_structure(lambda x: tf.gather(x, action_idx), traj['action'])
    traj['task']['delta_time'] = tf.cast(action_idx - tf.range(traj_len), tf.float32) / freq
    
    return traj