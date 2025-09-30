import numpy as np
import os 
import pickle
from pathlib import Path
import natsort

class lazy_loader:
    def __init__(self, memmap_list):
        self.memmap_list = memmap_list
        self.length = sum([len(memmap) for memmap in memmap_list])
        self.length_list = [len(memmap) for memmap in memmap_list]
        self.dim = memmap_list[0].shape[1]
        self.__create_mapping__()
        self.shape = (self.length, self.dim)
    
    def __create_mapping__(self):
        self.mapping = np.zeros(self.length, dtype=int)
        for i, length in enumerate(self.length_list):
            start_idx = sum(self.length_list[:i])
            self.mapping[start_idx:start_idx+length] = i
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, slices):
        # if is a single index
        if isinstance(slices, int):
            num_before = sum(self.length_list[:self.mapping[slices]])
            return self.memmap_list[self.mapping[slices]][slices - num_before]
        # if is a slice
        indices = self.mapping[slices]
        np_slices = np.arange(self.length)[slices]
        split_items = []
        for i in range(len(self.memmap_list)):
            split_indices = np_slices[indices == i] 
            split_indices -= sum(self.length_list[:i])
            split_item = self.memmap_list[i][split_indices]
            split_items.append(split_item)
        
        ret = np.concatenate(split_items, axis=0)
        
        return ret
    
class EmbeddingLoader:
    def __init__(self, embedding_foler: str, modalities: dict):
        self.embedding_path = []
        for root, dirs, files in os.walk(embedding_foler):
            for file in files:
                if file.endswith('features.pkl'):
                    self.embedding_path.append(os.path.join(root, file))
        self.embedding_path = natsort.natsorted(self.embedding_path)
                
        self.modalities = modalities
        
    def load_embeddings(self, concat=True, lazy_load=False, normalization=False) -> np.array:
        """
        Load embeddings from the embedding folder with features.pkl files. Multiple features.pkl files will be concatenated.
        Also, mutliple modalities can be concatenated as well.
        Returns:
            concat_embeddings: np.array, concatenated embeddings from all the features.pkl files
            metadata: dict, metadata for the embeddings
        """
        self.files = []
        metadata = dict()
        metadata['sequence'] = []
        metadata['modalities'] = self.modalities
        metadata['id'] = dict()
        metadata['id']['all'] = []
        for i in range(len(self.embedding_path)):
            # print(f'Loading embeddings from {self.embedding_path[i]}')
            with open(self.embedding_path[i], 'rb') as f:
                file = pickle.load(f)
                for modality in self.modalities.keys():
                    if modality not in file:
                        dat_path = Path(self.embedding_path[i].replace('features.pkl', f'{modality}.dat'))
                        assert dat_path.exists(), f'{dat_path} does not exist and {modality} is not in the features.pkl file'
                        dat = np.memmap(dat_path, dtype='float32', mode='r').reshape(file['id'].shape[0], -1)
                        file[modality] = dat
                self.files.append(file)
                dataset_name = Path(self.embedding_path[i]).parent.name
                metadata['sequence'].append((dataset_name, len(file[next(iter(self.modalities.keys()))])))
                metadata['id'][dataset_name] = file['id']
                metadata['id']['all'].append(file['id'])
        metadata['id']['all'] = np.concatenate(metadata['id']['all'], axis=0)
        embeddings = dict()
        num_traj = None
        
        scale_sum = sum([scale for modality, scale in self.modalities.items()])
        assert scale_sum == 1.0, 'The scale for all modalities should sum up to 1'
        
        def normalize(embeddings):
            if not lazy_load:
                embeddings = embeddings + 1e-6
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
                embeddings *= np.sqrt(scale)
            else:
                for i in range(len(embeddings.memmap_list)):
                    embeddings.memmap_list[i] = embeddings.memmap_list[i] + 1e-6
                    embeddings.memmap_list[i] /= np.linalg.norm(embeddings.memmap_list[i], axis=-1, keepdims=True)
                    embeddings.memmap_list[i] *= np.sqrt(scale)
                    
            return embeddings
        
        def check_shape(embeddings):
            num_traj = None
            for modality, scale in self.modalities.items():
                if num_traj is None:
                    num_traj = embeddings[modality].shape[0]
                else:
                    assert num_traj == embeddings[modality].shape[0], 'Number of trajectories should be the same for all modalities'

        for modality, scale in self.modalities.items():
            if not lazy_load:
                embeddings[modality] = np.concatenate([file[modality] for file in self.files], axis=0)
                embeddings[modality] = embeddings[modality].reshape(embeddings[modality].shape[0], -1)
            else:
                embeddings[modality] = lazy_loader([file[modality].reshape(file[modality].shape[0], -1) for file in self.files])
            if normalization:
                embeddings[modality] = normalize(embeddings[modality])
                
                
        check_shape(embeddings)
        concat_embeddings = {modality: embeddings[modality] for modality in self.modalities}
        if concat:
            concat_embeddings = np.concatenate([embeddings[modality] for modality in self.modalities], axis=1)
        
        return concat_embeddings, metadata

if __name__ == '__main__':
    embedding_folder = 'embeddings/rtx_chunked'
    modalities = {'image_embeds':0.8,'action':0.2}
    embedding_loader = EmbeddingLoader(embedding_folder, modalities)
    embeddings, meta = embedding_loader.load_embeddings()
    print(embeddings.shape)
                
                