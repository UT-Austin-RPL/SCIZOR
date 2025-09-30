import h5py

source_path = ""

with h5py.File(source_path, 'r+') as file:
    if 'mask' not in file.keys():
        round_idx = {'round0': [], 'round1': [], 'round2': [], 'round3': []}
        for demo_idx in file['data'].keys():
            if 'round' not in file[f'data/{demo_idx}'].keys():
                round_id = 3
            else:
                round_id = file[f'data/{demo_idx}/round'][0]
                if round_id == 1 and file['data'][demo_idx]['action_modes'][0] == -1:
                    round_id = 0
            round_idx[f'round{round_id}'].append(demo_idx)
        file.create_group('mask')
    else:
        round_idx = {f'round{i}': file['mask'][f'round{i}'][:].tolist() for i in range(4)}
    round_idx['round01'] = round_idx['round0'] + round_idx['round1']
    round_idx['round012'] = round_idx['round01'] + round_idx['round2']
    for round_id in round_idx.keys():
        if round_id not in file['mask'].keys():
            file['mask'].create_dataset(round_id, data=round_idx[round_id])
    print("Done!")