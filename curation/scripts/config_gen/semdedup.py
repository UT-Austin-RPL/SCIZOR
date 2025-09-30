from curation.scripts.config_gen.base import ConfigGenerator
import argparse
import time
import os
import yaml
from pathlib import Path

MODALITY_MAP = {
    "IM": ["image_embeds"],
    "AC": ["action"],
    "PR": ["proprio"],
    "VAE_MU": ["vae_mu"],
    "VAE_LOGVAR": ["vae_logvar"],
}

EPS_LIST = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.07, 0.1]

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

class SemDedupConfigGenerator(ConfigGenerator):
    def __init__(self):
        self.excutables = [
            f"cd {Path(__file__).parent.parent.parent.parent}",
            "conda activate curation",
            "python ./curation/semadedup/clustering.py --config-file $CONFIG_PATH --ngpus $NGPUS",
            "python ./curation/semadedup/sort_clusters.py --config-file $CONFIG_PATH",
            "python ./curation/semadedup/semdedup.py --config-file $CONFIG_PATH",
            "python ./curation/semadedup/concat_cluster_df.py --config-file $CONFIG_PATH",
        ]
        
    def _process_modality_list(self, value:list[str]):
        def process_modality(modality:str):
            modality_dict = dict()
            modalities = modality.split("-")
            if len(modalities) == 1:
                modality_code = modality[:-2]
                assert modality_code in MODALITY_MAP, f"Invalid modality, expected one of {MODALITY_MAP.keys()}, got {modality_code}"
                for modality_key in MODALITY_MAP[modality_code]:
                    modality_dict[modality_key] = 1.0/len(MODALITY_MAP[modality_code])
                return modality_dict
            for modality in modalities:
                modality_code = modality[:-2]
                assert modality_code in MODALITY_MAP, f"Invalid modality, expected one of {MODALITY_MAP.keys()}, got {modality_code}"
                modality_weight = int(modality[-2:])
                for modality_key in MODALITY_MAP[modality_code]:
                    modality_dict[modality_key] = modality_weight/100/len(MODALITY_MAP[modality_code])
            assert sum(modality_dict.values()) == 1, f"Sum of modality weights should be 1, got weights {modality_dict.values()}"
            return modality_dict
        processed_value = []
        for i in range(len(value)):
            processed_value.append(process_modality(value[i]))
        return processed_value
        
       
    def generate_config(self): 
        def dfs(config, ablation_configs, index):
            if index == len(ablation_configs):
                self.configs.append(config)
                return
            key = sorted(list(ablation_configs.keys()))[index]
            for value in ablation_configs[key]:
                config = config.copy()
                config[key] = value
                dfs(config, ablation_configs, index + 1)
        
        def add_eps_list():
            for config in self.configs:
                config["eps_list"] = EPS_LIST
                
        def add_timestamp():
            if self.args.timestamp is None:
                self.args.timestamp = time.strftime("%Y%m%d-%H%M%S")
            else:
                self.args.timestamp = self.args.timestamp + "-" + time.strftime("%Y%m%d-%H%M%S")
                
            for i, config in enumerate(self.configs):
                config["timestamp"] = self.args.timestamp + f"-{i}"
                
        self.configs:list[dict] = []
        self.ablation_configs: dict[list] = dict()
        common_config = dict()
        for arg in vars(self.args):
            if "octo" in arg:
                continue
            value = getattr(self.args, arg)
            if isinstance(value, list):
                if arg == "modality":
                    value = self._process_modality_list(value)
                self.ablation_configs[arg] = value
                common_config[arg] = None
            else:
                common_config[arg] = value
        dfs(common_config, self.ablation_configs, 0)
        add_eps_list()
        add_timestamp()
        
    def save_config(self, save_dir=os.path.join(os.path.dirname(__file__), "exps")):
        self.config_path = []
        save_dir = os.path.join(save_dir, self.args.ablation_name + "_" + self.args.timestamp)
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, config in enumerate(self.configs):
            folder = os.path.join(save_dir, config["timestamp"])
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(os.path.join(folder, "config.yaml"), "w") as fout:
                yaml.dump(config, fout)
                self.config_path.append(os.path.join(folder, "config.yaml"))
                
    def generate_script(self):
        self.scripts = []
        print("\n\n\n-----------------------------------------------------------------------------")
        assert self.config_path is not None, "Config should be saved before generating script"
        print("Total number of semdedup scripts:", len(self.config_path))
        print(f"Scripts and config paths: {self.save_dir}")
        print()
        assert (len(self.config_path) > 1 and len(self.args.octo_eps) == 1 and len(self.args.octo_random_curate_eps) == 0)\
            or len(self.config_path) == 1, "only one config can be ablated at a time"

        for i, config_path in enumerate(self.config_path):
            current_script = f"CONFIG_PATH={config_path}\n" + f"NGPUS={self.args.ngpus}\n"
            current_script += "\n".join(self.excutables)
            current_config = self.configs[i]
            self.scripts.append(current_script)
            save_dir = os.path.dirname(config_path)

            if len(self.args.octo_eps) == 1 and self.args.octo_command is not None:
                current_script += "\n\nconda activate octo\n"
                current_script += "\nexport WANDB_API_KEY=" + self.args.wandb_api + "\n"
                current_script += f"cd {Path(__file__).parent.parent.parent.parent}\n"
                current_script += "cd octo\n"
                current_save_path = os.path.join(self.args.save_folder, current_config["timestamp"])
                current_script += f"{self.args.octo_command} \\\n --config.curation.save_folder={current_save_path} \\\n"+\
                    f" --config.curation.curation_eps={self.args.octo_eps[0]}"
                
            with open(os.path.join(save_dir, f"semdedup.sh"), "w") as fout:
                fout.write(current_script)
                fout.write("\n\n")
                print("bash -i", os.path.join(save_dir, f"semdedup.sh"))
                print()
        
        octo_tag = self.args.octo_tag
        if self.args.octo_command is not None and (len(self.args.octo_eps) > 1 or len(self.args.octo_random_curate_eps) > 1):
            save_dir = os.path.dirname(self.config_path[0])
            print("Total number of octo scripts:", len(self.args.octo_eps) + len(self.args.octo_random_curate_eps))
            print()
            for i, eps in enumerate(self.args.octo_eps):
                current_script = "\n\nconda activate octo\n"
                current_script += "\nexport WANDB_API_KEY=" + self.args.wandb_api + "\n"
                current_script += "cd ./octo\n"
                current_save_path = os.path.join(self.args.save_folder, current_config["timestamp"])
                current_script += f"{self.args.octo_command} \\\n --config.curation.save_folder={current_save_path} \\\n"+ \
                    f" --config.curation.curation_eps={eps} \\\n --name {octo_tag}_octo_{eps}"
                with open(os.path.join(save_dir, f"{octo_tag}_octo_{eps}.sh"), "w") as fout:
                    fout.write(current_script)
                    fout.write("\n\n")
                    print("bash -i", os.path.join(save_dir, f"{octo_tag}_octo_{eps}.sh\n"))
            
            print()
                    
            for i, eps in enumerate(self.args.octo_random_curate_eps):
                current_script = "\n\nconda activate octo\n"
                current_script += "\nexport WANDB_API_KEY=" + self.args.wandb_api + "\n"
                current_script += "cd ./octo\n"
                current_save_path = os.path.join(self.args.save_folder, current_config["timestamp"])
                current_script += f"{self.args.octo_command} \\\n --config.curation.save_folder={current_save_path} \\\n"+ \
                    f" --config.curation.curation_eps={eps} \\\n--config.curation.random_curate=True\\\n --name {octo_tag}_octo_random_{eps}"
                with open(os.path.join(save_dir, f"{octo_tag}_octo_random_{eps}.sh"), "w") as fout:
                    fout.write(current_script)
                    fout.write("\n\n")
                    print("bash -i", os.path.join(save_dir, f"{octo_tag}_octo_random_{eps}.sh\n"))
                    
            if self.args.octo_add_control_group:
                current_script = "\n\nconda activate octo\n"
                current_script += "\nexport WANDB_API_KEY=" + self.args.wandb_api + "\n"
                current_script += "cd ./octo\n"
                current_save_path = os.path.join(self.args.save_folder, current_config["timestamp"])
                current_script += f"{self.args.octo_command}\\\n --name {octo_tag}_octo_control_group"
                with open(os.path.join(save_dir, f"{octo_tag}_octo_control_group.sh"), "w") as fout:
                    fout.write(current_script)
                    fout.write("\n\n")
                    print("bash -i", os.path.join(save_dir, f"{octo_tag}_octo_control_group.sh\n"))

                
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Octo SemDedup Config Generator")
        # SemDedup arguments
        parser.add_argument("--ablation_name", type=str, default="test", help="Name of the ablation study")
        parser.add_argument("--use_config", type=str, default=None, help="Use a config file")
        parser.add_argument("--emb_memory_folder", type=str, default=None, help="Folder to save video embeddings")
        parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the output")
        parser.add_argument("--niter", type=int, default=100, help="Number of iterations")
        parser.add_argument("--seed", type=int, default=1234, help="Random seed")
        parser.add_argument("--timestamp", type=str, default=None, help="save folder stamp")
        parser.add_argument("--ngpus", type=int, default=8, help="Number of GPUs")
        
        # Ablation arguments
        parser.add_argument("--ncentroids", nargs="+", type=int, default=[5000], help="Number of centroids")
        parser.add_argument("--Kmeans_with_cos_dist", nargs="+", type=str2bool, default=[True], help="Kmeans with cosine distance")
        parser.add_argument("--keep_hard", nargs="+", type=str2bool, default=[True], help="Keep hard examples")
        parser.add_argument("--which_to_keep", nargs="+", type=str, default=["random"], help="Which examples to keep", choices=["hard", "easy", "random"])
        parser.add_argument("--sim_metric", nargs="+", type=str, default=["cosine"], help="Similarity metric", choices=["cosine", "l2"])
        parser.add_argument("--modality", nargs="+", type=str, default=["IM80-AC20"], help="Format by Xi-Yj-Zk, where XYZ are modalities and ijk are the weights, summing to 100")
        # Modality: IM-Image, AC-Action, PR-Proprio    
        
        # Octo
        parser.add_argument("--octo_command", type=str, default=None, help="Run Octo training")
        parser.add_argument("--octo_eps", nargs="+", type=float, default=[0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07], help="episilon used for octo")
        parser.add_argument("--octo_random_curate_eps", nargs="+", type=float, default=[], help="random curate epsilon")
        parser.add_argument("--octo_add_control_group", action="store_true", help="Add control group")
        parser.add_argument("--octo_tag", type=str, default="", help="Tag for octo script")
        parser.add_argument("--wandb_api", type=str, default=None, help="Wandb API key")

        self.args = parser.parse_args() 
        assert (self.args.use_config is not None) or (self.args.emb_memory_folder is not None and self.args.save_folder is not None), "Either use a config file or provide emb_memory_folder and save_folder"  
        if self.args.octo_command is not None:
            assert self.args.wandb_api is not None, "save_folder and wandb_api should be provided for octo"
            if self.args.save_folder is None:
                assert self.args.use_config is not None, "save_folder should be provided for octo"
        
if __name__ == "__main__":
    config_generator = SemDedupConfigGenerator()
    config_generator.parse_args()
    if config_generator.args.use_config is None:
        config_generator.generate_config()
        config_generator.save_config()
    else:
        config_generator.config_path = [config_generator.args.use_config]
        config_path = config_generator.args.use_config if config_generator.args.use_config.endswith(".yaml") else os.path.join(config_generator.args.use_config, "config.yaml")
        config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        config_generator.configs = [config]
        config_generator.save_dir = config.get("save_folder")
        config_generator.args.save_folder = config.get("save_folder")
        assert config_generator.save_dir is not None, "save_folder should be provided in the config file"
    config_generator.generate_script()
    