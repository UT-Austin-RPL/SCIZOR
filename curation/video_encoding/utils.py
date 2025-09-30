from octo.octo.data.utils.data_utils import get_dataset_statistics, tree_map, normalize_action_and_proprio
from octo.octo.utils.spec import ModuleSpec
import dlimp as dl
from curation.suboptimal_classifier.dataset.rlds.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS, OXE_DATASET_CONTROL_FREQUENCY
from curation.suboptimal_classifier.dataset.rlds.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
import numpy as np

def get_statistics(name, builder):
    dataset = dl.DLataset.from_rlds(builder, split='all', shuffle=False)
    dataset = dataset.traj_map(OXE_STANDARDIZATION_TRANSFORMS[name])
    transform_spec = ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])
    proprio_obs_key = 'proprio'
    filter_function = ()
    
    dataset_statistics = get_dataset_statistics(
            dataset,
            hash_dependencies=(
                str(builder.info),
                str(proprio_obs_key),
                ModuleSpec.to_string(transform_spec).replace("curation.suboptimal_classifier.dataset.rlds.oxe","octo.data.oxe")
                if transform_spec is not None
                else "",
                *map(ModuleSpec.to_string, filter_function)
            ),
            save_dir=builder.data_dir,
            force_recompute=False,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)
    # action_statistics = dataset_statistics['action']
    return dataset_statistics    