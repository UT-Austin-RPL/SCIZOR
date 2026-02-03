# SCIZOR: Self-Supervised Data Curation for Large-Scale Imitation Learning

<!-- add a image of the scizor logo -->

<img src="images/scizor.png" alt="SCIZOR logo" style="width: 100%;">

SCIZOR is a self-supervised data curation framework that removes suboptimal and redundant data from large-scale datasets and enhances imitation learning policy performance.

[\[Paper\]](https://arxiv.org/pdf/2505.22626) [\[Project Page\]](https://ut-austin-rpl.github.io/SCIZOR/)
## Create curation env
```
mamba create -n curation python==3.11.0
mamba activate curation
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 faiss-gpu=1.8.0 -c pytorch -c nvidia
cd data-curation
pip install -r requirements.txt

cd LanguageBind
pip install -r requirements.txt
pip install -e .
cd ../

pip install -e ./dlimp
pip install -e ./data-curation
```

If encounter the following error:
```ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'```

please rename 
```from torchvision.transforms.functional_tensor``` to ```from torchvision.transforms._functional_tensor```
where the error occurs.

## Create customized Octo env with curation training
Install octo:
```
conda create -n octo python=3.10.1
conda activate octo
python -m pip install tensorflow[and-cuda]==2.14.0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

conda install cudnn=8.9 cuda-version=11.8
pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd octo
pip install -e .
pip install -r requirements.txt
cd ../
pip install -e ./dlimp
```
After the octo installation above success, you can test the octo installation by running the following command:
```
cd octo
python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --debug
```

## Create Customized Robomimic env with curation training
Install the customized Robomimic:
```
conda create -n robomimic python=3.7.12
conda activate robomimic
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
cd robomimic
pip install -e .
```


# Quick Start for Curation training on Octo
## Running Suboptimal
First fill in "config.dataset_kwargs.oxe_kwargs.data_dir" in the script below with the path to your OXE dataset folder, then run the following command to train the suboptimal classifier on OXE dataset.
``` 
conda activate curation
bash curation/scripts/subop_train_oxe.sh
```
After this, you can run the computing script to get the suboptimal scores for all transitions in the OXE dataset.
```
bash curation/video_encoding/get_all_score.sh <data_dir> <model_path> <output_dir> <batch_size> 
```

## Runing Dedup 
First, you need to download the cosmos model and encode the video in the OXE dataset into embeddings. Please follow the instructions below to encode the video:
```
bash curation/video_encoding/encode_all.sh <DATA_DIR> <OUTPUT_DIR> <MODEL_CACHE_DIR>
```

Run the following script to generate the dedup running script and launch the generated script.
```
python curation/scripts/config_gen/semdedup.py --ablation_name oxe --emb_memory_folder ~/embeddings/oxe --save_folder ~/expdata/oxe_semdedup
```


## Train the Octo Policy with curated dataset
You can run the following command on a multinode machine to train the curated octo
```
python scripts/multinode_train.py --config scripts/configs/octo_pretrain_config_curation.py:vit_s --config.dataset_kwargs.oxe_kwargs.data_dir <OXE_DATA_DIR>  --config.save_dir <PATH_TO_SAVE_CKPT> --config.dataset_kwargs.oxe_kwargs.data_mix=oxe_magic_soup --config.viz_kwargs.eval_batch_size=128 --config.save_interval=10000 --config.dataset_kwargs.batch_size=2048 --config.num_steps=300000 --config.optimizer.learning_rate.peak_value=3e-4 --config.wandb.project octo --config.dataset_kwargs.shuffle_buffer_size=200000 --name=octo_curation --config.curation.subop_score_path <PATH_TO_SUBOP_SAVE_DIR> --config.curation.subop_score_keep_percentile 0.881  --config.curation.save_folder <PATH_TO_DEDUP_SAVE_DIR> --config.curation.curation_eps=0.955 --config.curation.per_dataset_subop_thres=True --config.curation.subop_mix_level=0.5
```


# Quick Start for Curation training on Robomimic
## Running Suboptimal
First fill in "config.hdf5_dataset_kwargs.data_dir" in the script below with the path to your Robomimic dataset folder, then train the suboptimal classifier on Robomimic dataset by running the following command:
```
bash curation/scripts/subop_train_robomimic.sh
```
After this, you can run the computing script to get the suboptimal scores for all transitions in the Robomimic dataset.
```
python suboptimal_hdf5.py --data_dir <path_to_robomimic_data> --model_path <path_to_your_trained_suboptimal_model> --goal_time 2 --save_score
```

## Runing Dedup 
First, you need to download the cosmos model and encode the video in the Robomimic dataset into embeddings. Please follow the instructions below to encode the video:
```
python curation/video_encoding/video_encode_cosmos_hdf5.py --data_dir <path_to_robomimic_data> --output_dir <path_to_save_embeddings> --cosmos_path <path_to_cosmos_model>
```

Run the following script to generate the dedup running script and launch the generated script.
```
python curation/scripts/config_gen/semdedup.py --ablation_name robomimic --emb_memory_folder ~/embeddings/robomimic --save_folder ~/expdata/robomimic_semdedup
```

Write the computed result into the hdf5 file:
```
python curation/video_encoding/write_semdup_score.py --data_dir <path_to_robomimic_data> --semdup_path <path_to_your_dedup_result> 
```

## Train the Robomimic Policy with curated dataset
To launch the training, you need to first edit the "data" argument in the config file: `robomimic/exps/curation_exps/robomimic/square/bc_curation.json` and `robomimic/robomimic/exps/curation_exps/robomimic/can/bc_curation.json` to point to your curated Robomimic dataset, then you can run the following command under the Data-Curation-Robomimic directory:
```
python robomimic/scripts/train.py --config robomimic/exps/curation_exps/robomimic/can/bc_curation.json
python robomimic/scripts/train.py --config robomimic/exps/curation_exps/robomimic/square/bc_curation.json
```


## BibTeX

If you find this work useful, please cite it as follows:

```bibtex
@inproceedings{zhang2026scizor,
  title={SCIZOR: Self-Supervised Data Curation for Large-Scale Imitation Learning},
  author={Zhang, Yu and Xie, Yuqi and Liu, Huihan and Shah, Rutav and Wan, Michael and Fan, Linxi and Zhu, Yuke},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
