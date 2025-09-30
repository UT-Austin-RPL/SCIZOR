from ml_collections.config_dict import FieldReference, placeholder
from ml_collections import ConfigDict

PI = 3.1415926

def get_config():
    window_size = FieldReference(default=1)
    action_horizon = FieldReference(default=1)
    future_action = FieldReference(default=0)
    future_image = FieldReference(default=False)
    num_steps = FieldReference(default=50000)
    lr = FieldReference(default=2e-5)
    time_bins=[[0.0, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 5.0], [5.0, 10000.0]]
    
    return ConfigDict(
        dict(
            seed=42,
            window_size=window_size,
            action_horizon=action_horizon,
            future_action=future_action,
            future_image=future_image,
            num_datasets=1,
            load_path=placeholder(str),
            discriminator_dataset_kwargs = dict(
                action_aug_kwargs = dict(
                    flip_arg=dict(
                        randomize_prob=0.05,
                        grip_flip_score=2.0,
                    ),
                    transform_xyz_arg=dict(
                        randomize_prob=0.25,
                        rotation_kwargs=dict(
                            strategy='normal',
                            theta_range=(PI/2, PI),
                            # Below are only used if strategy='normal'
                            mean_theta=2*PI/3,
                            std_theta=PI/6,
                        ),
                        rotation_score_scale=4.0,
                    ),
                    random_xyz_rpy_scale_arg=dict(
                        randomize_prob=0.1,
                        xyz_rpy_norm_score_scale = 2.0,
                        xyz_rpy_score_alpha=5.0,
                        strategy='bimodal',
                        random_scale=dict(
                            scale_range=(0.0, 8.0),
                            # Below are only used if strategy='bimodal'
                            peaks=(0.05, 3.0), 
                            std = (0.05, 0.5),
                            peak_prob=0.6,
                        )
                    ),
                    random_flip_rpy_arg=dict(
                        randomize_prob=0.1,
                        rpy_action_scale=0.5,
                        rpy_flip_score_scale=1.5,
                    ),
                    score_range=(0.0, 2.0),
                ),

                obs_aug_kwargs = dict(
                    dropout_image_prob=0.1,
                    text_add_front_prob=0.5,
                    text_front = "robot arm . gripper . "
                ),
                image_key="image_primary",
                proprio_noise_scale=0.002,
                action_noise_scale=0.02,
                history_aug_prob=0.25,
                action_aug=False,
            ),
            
            subop_dataset = dict(
                path = placeholder(str),
                metrics = 'fail_grasp,',
                dataset_names = placeholder(str),
                batch_size = 1,
                label_balance = False,
            ),  
            
            dali_kwargs=dict(
                device="cpu",
                num_dataset=1,
                num_device=1,
                prefetch_queue_depth=2,
                save_gpu_memory=False,
                num_threads_per_dataset=32,
                buffer_device=placeholder(str),
                buffer_size=2,
            ),
            dataset_kwargs = get_dataset_config(window_size=window_size, time_bins=time_bins, action_horizon=action_horizon),
            hdf5_dataset_kwargs = dict(
                data_dir=placeholder(str),
                obs_keys=dict(agentview_image=84, robot0_eye_in_hand_image=84),
                freq=20,
                batch_size=32,
                time_bins=time_bins,
                num_workers=4,
            ),
                
            
            prefetch_num_batches=2,
            
            curation = dict(
                curation_df_path=placeholder(str),
                save_folder=placeholder(str),
                curation_eps=0.02,
                semdedup_config=dict(),
                random_curate=False,
                
                pivot_df_path=placeholder(str),
                pivot_start_weight=1.0,
                pivot_end_weight=3.0,
                pivot_start_step=10000,
                pivot_end_step=200000, 
                
                base_control_freq = placeholder(int),
                curate_small_action_thres = 0.0,
                curate_short_traj_len = 0,
                
                hardcode_subop_curation = False,
                load_delta_proprio=True,
                load_delta_gripping_act=True,
                delta_gripping_act_ratio=0.0,
            ),
            
            discriminator = dict(
                model_cfg_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                weights_path="GroundingDINO/weights/groundingdino_swint_ogc.pth",
                head_type="regression",
                head_token="cls",
                loss_fn_type="mse",
                action_query_length=64,
                no_action_input=False,
                no_text_input=False,
                frozen_encoder=True,
                fusion_blocks_type='cross-attn',
                encoder_type='groundingdino',
                use_norm_delta_proprio=False,
                use_delta_grip_act=False,
                num_ranks=placeholder(int),
                rank_thres = time_bins,
                window_size=window_size,
                action_horizon=action_horizon,
                future_action=future_action,
                future_image=future_image,
                histroy_dropout_rate=0.25,
                num_blocks=12,
                d_model=256,
                n_heads=8,
                normalize_linear_combination=False,
                activate_action_embedding=True,
                cat_start_feature=False,
                metrics=dict(
                    mse = True,
                    mae = True,
                    thres = [0.5, 0.6, 0.8],
                    pred_dist_bins = 5,
                ),
                eval_metrics=dict(
                    mse = False,
                    mae = False,
                    thres = [0.5, 0.6, 0.7, 0.8],
                    pred_dist_bins = 3,
                ),
            ),
            
            optimizer = dict(
                lr=lr,
                weight_decay=1e-3,
            ),
            
            scheduler = dict(
                # OneCycleLR
                max_lr=lr,
                total_steps=num_steps,
                pct_start=0.1,
            ),
            num_steps=num_steps,
            grad_accum_steps=1,
            eval_interval=2000,
            save_interval=10000,
            log_interval=100,
            eval_hdf5_path=placeholder(str),
            eval_batch_size=32,
            save_dir=placeholder(str),
            
            wandb=dict(
                project="suboptimal_classifier",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            mixed_precision=False,
        )
    )

def get_dataset_config(window_size=1, time_bins=None, action_horizon=1):
    task_augmentation = dict(
        task_augment_strategy=placeholder(str),
        task_augment_kwargs=dict(
            keep_image_prob=0.5,
        ),
    )

    return dict(
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        oxe_kwargs=dict(
            data_mix=placeholder(str),
            data_dir=placeholder(str),
            load_camera_views=("primary", "wrist"),
            load_depth=False,
        ),
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=action_horizon,
            goal_relabeling_strategy="custom_uniform_v4",
            goal_relabeling_kwargs=dict(
                max_goal_time_diff=placeholder(float),
                min_goal_time_diff=-1.0,
                time_bins=time_bins,
                neighbor_time=2.0,
            ),
            subsample_length=100,
            **task_augmentation,
        ),
        frame_transform_kwargs=dict(
            resize_size=dict(primary=(256, 256), wrist=(128, 128)),
            image_dropout_prob=0.0,
            image_augment_kwargs=dict(
                primary=dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.1],
                    augment_order=[
                        "random_resized_crop",
                        # "random_brightness",
                        # "random_contrast",
                        # "random_saturation",
                        # "random_hue",
                    ],
                )
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,  # shared between all datasets
        traj_read_threads=48,  # shared between all datasets
        shuffle_buffer_size=25000,  # shared between all datasets
        batch_size=512,
        balance_weights=True,
    )
