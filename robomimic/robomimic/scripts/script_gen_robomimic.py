import tyro
import os
import json
import sys
import robomimic

robomimic_path = os.path.dirname(robomimic.__file__)
def generate_multiple_exp(
    config_path:str=os.path.join(robomimic_path ,"exps/curation_exps"), 
    output_path:str=os.path.join(robomimic_path ,"../datasets/training_configs/curation_exps"),
    num_seeds:int=3, 
    subop_percentiles:list=[],
    dedup_percentiles:list=[],
    clean_all:bool=False,
    batched:bool=False,
    removed_keys:list=[],
):
    if batched:
        output_path = output_path.replace("curation_exps", "curation_exps_batched")

    if clean_all:
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith(".json") and ("seed" in file or "subop" in file):
                    os.remove(os.path.join(root, file))

    os.makedirs(output_path, exist_ok=True)
    assert num_seeds > 0, "num_seeds must be greater than 0"
    #walk through the config_path
    for root, dirs, files in os.walk(config_path):
        for file in files:
            if file.endswith(".json") and "seed" not in file:
                if any([key in file for key in removed_keys]):
                    continue
                for seed in range(1, num_seeds+1):
                    with open(os.path.join(root, file), "r") as f:
                        config = json.load(f)
                    config['train']["seed"] = seed
                    if batched:
                        config['experiment']['rollout']['batched'] = True
                        # config['experiment']['rollout']['num_batch_envs'] = config['experiment']['rollout']['n']
                    output_root = root.replace(config_path, output_path)
                    os.makedirs(output_root, exist_ok=True)
                    if len(subop_percentiles) > 0:
                        for ratio in subop_percentiles:
                            config['train']['curation']['subop_curate']['enabled'] = True
                            config['train']['curation']['subop_curate']['subop_percentile'] = float(ratio)
                            
                            output_dir = config['train']['output_dir']
                            split = output_dir.split("/")
                            if float(ratio) == 1.0:
                                config['train']['curation']['subop_curate']['enabled'] = False
                            if len(dedup_percentiles) > 0:
                                for dedup_ratio in dedup_percentiles:
                                    config['train']['curation']['dedup_curate']['enabled'] = True
                                    config['train']['curation']['dedup_curate']['keep_ratio'] = float(dedup_ratio)

                                    if float(dedup_ratio) == 1.0:
                                        config['train']['curation']['dedup_curate']['enabled'] = False
                                    
                                    for i in range(len(split)):
                                        if "subop_" in split[i]:
                                            split[i] = f"subop_{ratio}_dedup_{dedup_ratio}"
                                            if batched:
                                                split[i] = f"subop_{ratio}_dedup_{dedup_ratio}_batched"
                                    config['train']['output_dir'] = "/".join(split)
                                    config['train']['curation']['dedup_curate']['seed'] = seed
                                    with open(os.path.join(output_root, file.replace(".json", f"_subop_{ratio}_dedup_{dedup_ratio}_seed{seed}.json")), "w") as f:
                                        json.dump(config, f, indent=4)
                            else:
                                for i in range(len(split)):
                                    if "subop_" in split[i]:
                                        split[i] = f"subop_{ratio}"
                                        if batched:
                                            split[i] = f"subop_{ratio}_batched"
                                config['train']['output_dir'] = "/".join(split)
                                with open(os.path.join(output_root, file.replace(".json", f"_subop_{ratio}_seed{seed}.json")), "w") as f:
                                    json.dump(config, f, indent=4)
                    else:
                        with open(os.path.join(output_root, file.replace(".json", f"_seed{seed}.json")), "w") as f:
                            json.dump(config, f, indent=4)
                            
                
if __name__ == "__main__":
    tyro.cli(generate_multiple_exp)