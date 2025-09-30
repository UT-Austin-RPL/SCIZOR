import tensorflow as tf
import time

PI=3.1415926

def grip_flip(action: tf.Tensor,
              flip_prob: float,
              delta_grip_act: tf.Tensor,
            ):
        """
        Flip the gripping action
        """
        
        traj_len = action.shape[0]
        flip = tf.random.uniform([traj_len]) < flip_prob
        fliped_action = tf.where(tf.broadcast_to(flip[:, None, None], action.shape) , 1 - action, action)
        delta_grip_act = tf.where((flip&(fliped_action==0))[..., 0]&(delta_grip_act==0), -1.0, delta_grip_act)
        delta_grip_act = tf.where((flip&(fliped_action==1))[..., 0]&(delta_grip_act==0), 1.0, delta_grip_act)
        
        return fliped_action, flip, delta_grip_act
    
def rpy_flip(action: tf.Tensor,
             flip_prob: float,
            ):

    traj_len = action.shape[0]
    flip = tf.random.uniform([traj_len]) < flip_prob
    # action_flip = tf.repeat(flip[:, None, None], 3, axis=-1)
    action_flip = tf.broadcast_to(flip[:, None, None], action.shape)
    fliped_action = tf.where(action_flip, -action, action)

    return fliped_action, flip


def random_unit_vector(traj_len: int,):
    phi = tf.random.uniform([traj_len], minval=0, maxval=2*PI)
    costheta = tf.random.uniform([traj_len], minval=-1, maxval=1)
    theta = tf.acos(costheta)
    x = tf.sin(theta) * tf.cos(phi)
    y = tf.sin(theta) * tf.sin(phi)
    z = tf.cos(theta)
    xyz = tf.stack([x, y, z], axis=-1)
    return xyz

def random_rotation(traj_len: int,
                    original_xyz: tf.Tensor,
                    strategy: str,
                    theta_range: tuple[float, float]=(0, PI),
                    mean_theta: float=PI/2,
                    std_theta: float=PI/2,
):
    random_vec = random_unit_vector(traj_len)
    rotation_axis = tf.linalg.cross(random_vec, original_xyz[:, -1])
    rotation_axis = rotation_axis/tf.norm(rotation_axis, axis=-1)[:, None]
    if strategy == 'uniform':
        theta = tf.random.uniform([traj_len], minval=theta_range[0], maxval=theta_range[1])
    elif strategy == 'normal':
        theta = tf.random.normal([traj_len], mean=mean_theta, stddev=std_theta)
    theta = tf.clip_by_value(theta, theta_range[0], theta_range[1])
    K_matrix = tf.stack([tf.zeros_like(theta), -rotation_axis[:, 2], rotation_axis[:, 1],
                        rotation_axis[:, 2], tf.zeros_like(theta), -rotation_axis[:, 0],
                        -rotation_axis[:, 1], rotation_axis[:, 0], tf.zeros_like(theta)], axis=-1)
    K_matrix = tf.reshape(K_matrix, [-1, 3, 3])

    rotation_matrix = tf.eye(3)[None,...] + tf.sin(theta)[:, None, None] * K_matrix + (1 - tf.cos(theta)[:, None, None]) * tf.matmul(K_matrix, K_matrix)
    rotation_matrix = tf.repeat(rotation_matrix[:,None], tf.shape(original_xyz)[1], axis=1)
    rotated_xyz = tf.matmul(rotation_matrix, original_xyz[..., None])
    rotated_xyz = rotated_xyz[..., 0]
    return rotated_xyz, theta
    
def random_scale(xyz: tf.Tensor,
                 scale_range: tuple[float, float],
                 strategy: str='bimodal',
                 peaks: tuple[float, float]=(0.2, 3.0),
                 std: tuple[float, float]=(0.1, 0.5),
                 peak_prob: float = 0.5
                 ):
    """
    Randomly scale the action
    """
    if strategy == 'uniform':
        scale = tf.random.uniform([xyz.shape[0]], minval=scale_range[0], maxval=scale_range[1])
    elif strategy == 'bimodal':
        peak1, peak2 = peaks
        std1, std2 = std
        prob = tf.random.uniform([xyz.shape[0]])
        scale = tf.where(prob < peak_prob, tf.random.normal([xyz.shape[0]], mean=peak1, stddev=std1), tf.random.normal([xyz.shape[0]], mean=peak2, stddev=std2))
        scale = tf.clip_by_value(scale, scale_range[0], scale_range[1])
        
    xyz = xyz * scale[:, None, None]
    return xyz, scale
    
        
        
     
# @tf.function
def augment_action( action: tf.Tensor,
                    norm_delta_proprio: tf.Tensor,
                    delta_grip_act: tf.Tensor,
                    action_aug_kwargs: dict,
                   ):
    """
    Augment an action
    """
    assert action.shape[-1] == 7, 'Each action must have 7 dimensions'
    action_shape = tf.shape(action)
    norm_delta_proprio_shape = tf.shape(norm_delta_proprio)
    norm_delta_proprio = tf.convert_to_tensor(norm_delta_proprio)
    delta_grip_act = tf.convert_to_tensor(delta_grip_act)
    action = tf.convert_to_tensor(action)
    
    # if len(action_shape) == 3:
    #     action = tf.reshape(action, [-1, 7])
    #     norm_delta_proprio = tf.reshape(norm_delta_proprio, [-1, 1])
    assert len(tf.shape(action)) == 3 and len(tf.shape(norm_delta_proprio)) == 2 and len(tf.shape(delta_grip_act)) == 2, 'Action and norm_delta_proprio must have shape [traj_len, history, 7] and [traj_len, history] respectively'
        
    traj_len = action.shape[0]
    score = tf.zeros([traj_len], dtype=tf.float32)
    xyz, rpy, gripping = tf.split(action, [3, 3, 1], axis=-1)
    
    flip_arg = action_aug_kwargs['flip_arg']
    transform_xyz_arg = action_aug_kwargs['transform_xyz_arg']
    random_xyz_rpy_scale_arg = action_aug_kwargs['random_xyz_rpy_scale_arg']
    random_flip_rpy_arg = action_aug_kwargs['random_flip_rpy_arg']
    
    if flip_arg['randomize_prob'] > 0:
        gripping, flip, delta_grip_act = grip_flip(gripping, flip_arg['randomize_prob'], delta_grip_act)
        flip_score = tf.where(flip, flip_arg['grip_flip_score'], 0)
        score += flip_score
    else:
        flip_score = tf.zeros([traj_len], dtype=tf.float32)
        
    if transform_xyz_arg['randomize_prob'] > 0:
        transform_xyz_arg = action_aug_kwargs['transform_xyz_arg']
        to_rotate = tf.random.uniform([traj_len]) < transform_xyz_arg['randomize_prob']
        broadcasted_to_rotate = tf.broadcast_to(to_rotate[:, None, None], xyz.shape)
        rotated_xyz, theta = random_rotation(traj_len, xyz, **transform_xyz_arg['rotation_kwargs'])
        xyz = tf.where(broadcasted_to_rotate, rotated_xyz, xyz)
        angle_score = tf.where(to_rotate, theta/PI, 0)
        score += angle_score*transform_xyz_arg['rotation_score_scale']
    else:
        angle_score = tf.zeros([traj_len], dtype=tf.float32)
        
    if random_xyz_rpy_scale_arg['randomize_prob'] > 0:
        alpha = random_xyz_rpy_scale_arg['xyz_rpy_score_alpha']
        to_scale = tf.random.uniform([traj_len]) < random_xyz_rpy_scale_arg['randomize_prob']
        to_scale = to_scale & (norm_delta_proprio[:, -1] < 0.01)
        rand_scaled_xyz, scale = random_scale(xyz, **random_xyz_rpy_scale_arg['random_scale'])
        rand_scaled_rpy = rpy * scale[:, None, None]
        broadcasted_to_scale = tf.broadcast_to(to_scale[:, None, None], xyz.shape)
        xyz = tf.where(broadcasted_to_scale, rand_scaled_xyz, xyz)
        rpy = tf.where(broadcasted_to_scale, rand_scaled_rpy, rpy)
        norm_delta_proprio = tf.where(to_scale[:, None], norm_delta_proprio * scale[:, None], norm_delta_proprio)
        xyz_rpy_scale_score = tf.where(scale > 1, tf.math.pow((scale - 1)/alpha, alpha), 1/(tf.math.pow(scale, 1/alpha)+1e-5)-1)
        xyz_rpy_scale_score = tf.where(to_scale, xyz_rpy_scale_score, 0)
        xyz_rpy_scale_score = tf.clip_by_value(xyz_rpy_scale_score, 0, 1) * random_xyz_rpy_scale_arg['xyz_rpy_norm_score_scale']
        score += xyz_rpy_scale_score
    else:
        xyz_rpy_scale_score = tf.zeros([traj_len], dtype=tf.float32)
        
    if random_flip_rpy_arg['randomize_prob'] > 0:
        rpy, flip = rpy_flip(rpy, random_flip_rpy_arg['randomize_prob'])
        rpy_flip_score = tf.where(flip, 1.0, 0.0) * tf.norm(rpy[:, -1], axis=-1) * random_flip_rpy_arg['rpy_action_scale']
        rpy_flip_score = tf.clip_by_value(rpy_flip_score, 0.0, 1.0) * random_flip_rpy_arg['rpy_flip_score_scale']
        
        score += rpy_flip_score    
    else:
        rpy_flip_score = tf.zeros([traj_len], dtype=tf.float32)
        
    action = tf.concat([xyz, rpy, gripping], axis=-1)
    
    action = tf.reshape(action, action_shape)
    # score = tf.reshape(score, action_shape[:-1])
    norm_delta_proprio = tf.reshape(norm_delta_proprio, norm_delta_proprio_shape)
    
    score_range = action_aug_kwargs['score_range']
    score = tf.clip_by_value(score, score_range[0], score_range[1])/(score_range[1] - score_range[0])
    sub_score = tf.stack([flip_score, angle_score, xyz_rpy_scale_score, rpy_flip_score], axis=-1)
    sub_score = tf.reshape(sub_score, tf.concat([[traj_len], sub_score.shape[-1:]], axis=-1))
        
    return action, score, sub_score, norm_delta_proprio, delta_grip_act
        
        
        
if __name__ == '__main__':
    action = tf.random.uniform([1024, 6], minval=-1, maxval=1)
    grip_action = tf.random.uniform([1024, 1], minval=0, maxval=1)>0.5
    grip_action = tf.cast(grip_action, tf.float32)
    action = tf.concat([action, grip_action], axis=-1)
    
    action_aug_kwargs = dict(
        grip_flip=dict(
            randomize_prob=0.5,
            grip_flip_score=1.0,
        ),
        transform_xyz_direction=dict(
            randomize_prob=0.5,
            rotation_kwargs=dict(
                strategy='normal',
                theta_range=(0, PI),
                phi_range=(0, PI),
                mean_theta=PI/2,
                mean_phi=PI/2,
            ),
            rotation_score_scale=1.5,
        ),
        random_xyz_scale=dict(
            randomize_prob=0.5,
            scale_range=(0.5, 1.5),
            xyz_score_alpha=3.0,
        ),
        random_flip_ryz=dict(
            randomize_prob=0.1,
            ryz_flip_score_scale=1.0,
        ),
    )
    aug_action, score = augment_action(action, **action_aug_kwargs)
    start = time.time()
    scores = []
    for i in range(500):
        aug_action, score = augment_action(action, **action_aug_kwargs)
        scores.append(score)
    print(f"Average Time taken: {(time.time()-start)/100}")
    print(f"Average Score: {tf.reduce_mean(scores)}")
    print(f"Max Score: {tf.reduce_max(scores)}")
    print(f"Min Score: {tf.reduce_min(scores)}")
    print(f"Std Score: {tf.math.reduce_std(scores)}")
    print(f"Num of 0 scores: {tf.reduce_sum(tf.cast(tf.equal(scores, 0), tf.int32))}, Num of non-zero scores: {tf.reduce_sum(tf.cast(tf.not_equal(scores, 0), tf.int32))}")