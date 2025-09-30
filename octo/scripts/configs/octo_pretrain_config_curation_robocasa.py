from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader
from ml_collections.config_dict import placeholder


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    action_dim = FieldReference(7)
    act_score_cond = FieldReference(False)

    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
        "secondary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_secondary"],
            task_stack_keys=["image_secondary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
        "wrist": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_wrist"],
            task_stack_keys=["image_wrist"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }
    config["model"]["repeat_task_tokens"] = False
    config["model"]["readouts"] = {"action": 1}

    # We augment differently for the primary and wrist cameras
    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.9, 1.0], ratio=[1.0, 1.0]),
        random_brightness=[0.2],
        random_contrast=[0.8, 1.2],
        random_saturation=[0.8, 1.2],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    
    secondary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.9, 1.0], ratio=[1.0, 1.0]),
        random_brightness=[0.2],
        random_contrast=[0.8, 1.2],
        random_saturation=[0.8, 1.2],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    
    wrist_augment_kwargs = dict(
        random_brightness=[0.2],
        random_contrast=[0.8, 1.2],
        random_saturation=[0.8, 1.2],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (128, 128),  # workspace camera is at 256x256
        "secondary": (128, 128),  # workspace camera is at 256x256
        "wrist": (128, 128),  # wrist camera is at 128x128
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
        "secondary": secondary_augment_kwargs,
        "wrist": wrist_augment_kwargs,
    }

    config = update_config(
        config,
        num_steps=300000,
        save_start=0,
        window_size=2,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="oxe_magic_soup",
                data_dir="gs://rail-orca-central2/resize_256_256",
                load_camera_views=("primary", "secondary", "wrist"),
                load_depth=False,
                force_recompute_dataset_statistics=False,
            ),
            traj_transform_kwargs=dict(
                action_horizon=4,
                max_action_dim=action_dim,
                task_augment_strategy="delete_task_conditioning",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                ),
            ),
            batch_size=1024,
            shuffle_buffer_size=500000,
            balance_weights=True,
        ),
        curation = dict(
            curation_df_path=placeholder(str),
            save_folder=placeholder(str),
            curation_eps= "0.9",
            curation_prob="1.0",
            semdedup_config=dict(),
            random_curate=False,
            chunk_time=placeholder(float),
            curation_rebalance=False,
            
            pivot_df_path=placeholder(str),
            pivot_start_weight=1.0,
            pivot_end_weight=3.0,
            pivot_start_step=10000,
            pivot_end_step=200000, 
            
            base_control_freq = placeholder(int),
            curate_small_action_thres = 0.0,
            curate_short_traj_len = 0,
            
            hardcode_subop_curation = False,
            load_delta_proprio=False,
            
            subop_score_path = placeholder(str),
            subop_score_keep_percentile = 0.9,
            subop_score_start_curate_percentile = 1.0,
            subop_score_thres = placeholder(float),
            subop_score_start_thres = placeholder(float),
            subop_sampling = False,
            subop_score_type = 'suboptimal_score',
            average_time = 0.0,
            delta_time = 2.0,
            act_score_cond = act_score_cond,
            traj_level_subop = False,
            per_dataset_subop_thres = False,
            
            curate_before_sample = False,
        ),
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=["robocasa"],
        dali_kwargs=dict(
            use_dali=True,
            device="mixed",
            num_dataset=1,
            num_device=1,
            prefetch_queue_depth=2,
            save_gpu_memory=False,
            num_threads_per_dataset=32,
            buffer_device=placeholder(int),
        )
            
    )
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHead,
        readout_key="readout_action",
        use_map=False,
        action_horizon=4,
        action_dim=action_dim,
        n_diffusion_samples=1,
        dropout_rate=0.0,
        act_score_cond=act_score_cond,
    )

    return config
