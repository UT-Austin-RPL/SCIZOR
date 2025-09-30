import os
from pprint import pprint
import tyro
import numpy as np
import pandas as pd
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_seed_mean(results):
    mean_result = {}
    std = {}
    for path, result in results.items():
        base_exp = path.split('/')[:-2]
        base_exp = '/'.join(base_exp)
        if base_exp not in mean_result:
            mean_result[base_exp] = []
        mean_result[base_exp].append(result)

    for base_exp, results in mean_result.items():
        std[base_exp] = np.std(results)
        # assert len(results) == 3 or len(results) == 2, (base_exp, len(results))
        # mean_result[base_exp] = sum(results) / len(results)
        mean_result[base_exp] = np.mean(results, axis=0)
    return mean_result, std

def count_results(path:str, filter_keys:list=[], remove_keys:list=[], from_json:bool=False, top:int=3,):
    result = {}
    for root, dirs, files in os.walk(path):
        filtered = [key in root for key in filter_keys]
        if not all(filtered):
            continue
        removed = [key in root for key in remove_keys]
        if any(removed):
            continue
        if len(files) == 0:
            continue
        for file in files:
            if not from_json:
                if not file.endswith('.pth'):
                    continue
                name_split = file.split('_')
                if len(name_split) == 3:
                    continue
                success_rate = float(name_split[-1][:-4])
                if root not in result:
                    result[root] = []
                # else:
                    # result[root] = max(result[root], success_rate)
                result[root].append(success_rate)
            else:
                if not file.endswith('.json'):
                    continue
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    if root not in result:
                        # result[root] = data['Success_Rate']
                        result[root] = []
                    # else:
                    #     result[root] = max(result[root], data['Success_Rate'])
                    result[root].append(data['Success_Rate'])
    results = {}
    for k, r in result.items():
        # mean of top k result
        top_k = top
        assert len(r) >= 20, f"Checkpoint {k} has less than 20 results"
        results[k] = sum(sorted(r, reverse=True)[:top_k]) / top_k
    csv_file_name_prefix = "_".join(filter_keys) + f"_top{top_k}"
    return results, csv_file_name_prefix, result
    
def plot_grid_results(mean_result, csv_file_name_prefix):
    # Extract subop and dedup values from keys using regex
    subop_values = set()
    dedup_values = set()
    
    pattern = r'subop_([0-9.]+)_dedup_([0-9.]+)'
    
    # Process each key to extract parameter values
    extracted_data = {}
    for key, success_rate in mean_result.items():
        match = re.search(pattern, key)
        if match:
            subop = float(match.group(1))
            dedup = float(match.group(2))
            
            subop_values.add(subop)
            dedup_values.add(dedup)
            
            # Store the success rate for this parameter combination
            extracted_data[(dedup, subop)] = success_rate
    
    # Convert sets to sorted lists
    subop_values = sorted(list(subop_values), reverse=True)
    dedup_values = sorted(list(dedup_values), reverse=False)
    print(subop_values)
    print(dedup_values)
    
    # Create a matrix for the heatmap
    matrix = np.zeros((len(dedup_values), len(subop_values)))
    
    # Fill the matrix with success rates
    for i, dedup in enumerate(dedup_values):
        for j, subop in enumerate(subop_values):
            if (dedup, subop) in extracted_data:
                matrix[i, j] = extracted_data[(dedup, subop)]
            else:
                matrix[i, j] = np.nan  # Use NaN for missing values

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, 
        xticklabels=[f"{v*100}" for v in subop_values], 
        yticklabels=[f"{v*100}" for v in dedup_values], 
        cmap="YlGnBu", annot=True, fmt=".3f", 
        cbar_kws={'label': 'Success Rate (%)'}
    )

    # Add labels and title
    plt.xlabel('Subop Keep Percentage (%)')
    plt.ylabel('Dedup Keep Percentage (%)')
    plt.title(csv_file_name_prefix)

    # Show plot
    plt.savefig(f"plot_results/{csv_file_name_prefix}.png")

def plot_convergence(raw_results, csv_file_name_prefix):
    # first, create a new plot
    plt.figure()
    plt.title('Convergence Plot')
    for path, results in raw_results.items():
        # plot a curve for all the SR in results
        pattern = r'subop_([0-9.]+)_dedup_([0-9.]+)'
        # use pattern as the label
        match = re.search(pattern, path)
        if match:
            label = f"subop: {match.group(1)}, dedup: {match.group(2)}"
        else:
            label = path
        x = np.arange(len(results)) * 20
        results_increasing = [max(results[:i+1]) for i in range(len(results))]
        plt.plot(x, results_increasing, label=label)
    # put the legend under the figure
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
    plt.subplots_adjust(bottom=0.3)  
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate (%)')
    plt.savefig(f"plot_results/{csv_file_name_prefix}_convergence.png")


if __name__ == '__main__':
    result, csv_file_name_prefix, raw_results = tyro.cli(count_results)
    mean_result, std = calc_seed_mean(result)
    output_path = "./datasets/training_results/csv"
    all_result = pd.DataFrame(result.items(), columns=['path', 'success_rate'])
    all_result.to_csv(f'csv_results/{csv_file_name_prefix}_all_results.csv')
    mean_std_result = pd.DataFrame(mean_result.items(), columns=['path', 'mean_success_rate'])
    mean_std_result['std'] = std.values()
    mean_std_result.to_csv(f'csv_results/{csv_file_name_prefix}_mean_std_results.csv')

    if len(mean_result) > 10:
        plot_grid_results(mean_result, csv_file_name_prefix)
        print("save grid to", csv_file_name_prefix)

    # mean_raw_results, std_raw = calc_seed_mean(raw_results)
    # plot_convergence(mean_raw_results, csv_file_name_prefix)
    # else:
    pprint(result)
    print()
    print("Mean results:")
    pprint(mean_result)
    print()
    print("Standard deviation:")
    pprint(std)