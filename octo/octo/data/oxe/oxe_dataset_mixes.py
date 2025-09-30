"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""

GOOGLE_MIX = [
    ("fractal20220817_data", 1.0),
]

BRIDGE_MIX = [
    ("bridge_dataset", 1.0),
]

GOOGLE_BRIDGE_MIX = [
    ("fractal20220817_data", 1.0),
    ("bridge_dataset", 1.0),
]

RT_X_MIX = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 2.0),
    ("berkeley_cable_routing", 3.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 5.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
]


OXE_FRANKA_MIX = [
    ("taco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("viola", 1.0),
    ("toto", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("maniskill_dataset_converted_externally_to_rlds", 0.1),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("utaustin_mutex", 1.0),
    # ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
]


OXE_MAGIC_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
]

OXE_DIVERSITY_SOUP= [
    ("fractal20220817_data", 0.905),
    ("kuka", 0.56),
    ("bridge_dataset", 0.716),
    ("taco_play", 0.721),
    ("jaco_play", 0.922),
    ("berkeley_cable_routing", 0.938),
    ("roboturk", 0.872),
    ("nyu_door_opening_surprising_effectiveness", 0.413),
    ("viola", 0.50),
    ("berkeley_autolab_ur5", 0.366),
    ("toto", 0.174),
    ("language_table", 0.024),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 0.408),
    ("austin_buds_dataset_converted_externally_to_rlds", 0.515),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 0.177),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.398),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 0.154),
    ("austin_sailor_dataset_converted_externally_to_rlds", 0.882),
    ("austin_sirius_dataset_converted_externally_to_rlds", 0.276),
    ("bc_z", 0.499),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 0.419),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 0.323),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 0.577),
    ("berkeley_fanuc_manipulation", 0.506),
    ("cmu_stretch", 0.329),
]

OXE_FULL_SOUP = [
    ("fractal20220817_data", 1.0),
    ("kuka", 1.0),
    ("bridge_dataset", 1.0),
    ("taco_play", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
    ("language_table", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 1.0),
    ("cmu_stretch", 1.0),
]


OXE_FLEX_ACT_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    # ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
    ("gnm_dataset", 1.0),
    ("aloha_static_dataset", 3.0),
    # ("aloha_dagger_dataset", 1.0),
    ("aloha_mobile_dataset", 2.0),
    # ("fmb_dataset", 1.0),
    ("dobbe", 1.0),
    ("roboset", 0.5),
    ("rh20t", 0.5),
]


OXE_FULL_MIX = [
    ("fractal20220817_data", 1.0),
    ("kuka", 1.0),
    ("bridge_dataset", 1),
    ("taco_play", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
    ("language_table", 1.0),
    ("columbia_cairlab_pusht_real", 1.0),
    ("stanford_kuka_multimodal_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_rot_dataset_converted_externally_to_rlds", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("maniskill_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 1.0),
    ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", 1.0),
    ("utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_bimanual_converted_externally_to_rlds", 1.0),
    ("robo_net", 1.0),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 1.0),
    ("stanford_mask_vit_converted_externally_to_rlds", 1.0),
    ("tokyo_u_lsmo_converted_externally_to_rlds", 1.0),
    ("dlr_sara_pour_converted_externally_to_rlds", 1.0),
    ("dlr_sara_grid_clamp_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("asu_table_top_converted_externally_to_rlds", 1.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("imperialcollege_sawyer_wrist_cam", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("uiuc_d3field", 1.0),
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 1.0),
    ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
    ("cmu_stretch", 1.0),
    ("gnm_dataset", 1.0),
]

# mixes for re-mix
OXE_ALL = dict(
    rt1=dict(
        path="fractal20220817_data/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=3513684,
    ),
    kuka=dict(
        path="kuka/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=2133779,
    ),
    bridge=dict(
        path="bridge_dataset/1.0.0",
        train_split="train",
        val_split="val",
        weight=1946218,
    ),
    taco_play=dict(
        path="taco_play/0.1.0",
        train_split="train",
        val_split="test",
        weight=210730,
    ),
    taco_extra=dict(
        path="taco_extra/1.0.0",
        train_split="train[:95%]",
        val_split=None,  # Ignore this one since we have val data in taco_play
        weight=51756,
    ),
    jaco_play=dict(
        path="jaco_play/0.1.0",
        train_split="train",
        val_split="test",
        weight=69151,
    ),
    berkeley_cable_routing=dict(
        path="berkeley_cable_routing/0.1.0",
        train_split="train",
        val_split="test",
        weight=36758,
    ),
    roboturk=dict(
        path="roboturk/0.1.0",
        train_split="train",
        val_split="test",
        weight=166627,
    ),
    viola=dict(
        path="viola/0.1.0",
        train_split="train",
        val_split="test",
        weight=68778,
    ),
    berkeley_autolab_ur5=dict(
        path="berkeley_autolab_ur5/0.1.0",
        train_split="train",
        val_split="test",
        weight=86887,
    ),
    toto=dict(
        path="toto/0.1.0",
        train_split="train",
        val_split="test",
        weight=293237,
    ),
    language_table=dict(
        path="language_table/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=6275438,
    ),
    stanford_hydra=dict(
        path="stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=339237,
    ),
    austin_buds=dict(
        path="austin_buds_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=32757,
    ),
    furniture_bench=dict(
        path="furniture_bench_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=3739948,
    ),
    ucsd_kitchen=dict(
        path="ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=3614,
    ),
    austin_sailor=dict(
        path="austin_sailor_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=334011,
    ),
    austin_sirius=dict(
        path="austin_sirius_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=265812,
    ),
    bc_z=dict(
        path="bc_z/1.0.0",
        train_split="train",
        val_split="eval",
        weight=5432343,
    ),
    dlr_shared_control=dict(
        path="dlr_edan_shared_control_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=8487,
    ),
    iamlab_cmu_pickup_insert=dict(
        path="iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=138937,
    ),
    utaustin_mutex=dict(
        path="utaustin_mutex_resize/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=343702,
    ),
    berkeley_fanuc_manipulation=dict(
        path="berkeley_fanuc_manipulation_resize/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=58781,
    ),
    cmu_stretch=dict(
        path="cmu_stretch_resize/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=23556,
    ),
    #### the below datasets are NOT in the Octo Magic Soup Mix
    nyu_franka_play=dict(
        path="nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train",
        val_split="val",
        weight=34083,
    ),
    maniskill=dict(
        path="maniskill_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=4282929,
    ),
    cmu_franka_exploration=dict(
        path="cmu_franka_exploration_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=34083,
    ),
    kaist_nonprehensile=dict(
        path="kaist_nonprehensile_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        weight=30499,
    ),
    stanford_robocook=dict(
        path="stanford_robocook_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",

        weight=104863,
    ),
)

RTX_DOREMI_150K = {
    dataset: OXE_ALL[dataset] | dict(weight=weight)
    for dataset, weight in [
        ("berkeley_autolab_ur5", 0.02368840016424656),
        ("berkeley_cable_routing", 0.0019966200925409794),
        ("bridge", 0.1991041600704193),
        ("jaco_play", 0.0038572715129703283),
        ("kuka", 0.12065707892179489),
        ("roboturk", 0.011355401016771793),
        ("rt1", 0.3944754898548126),
        # ("taco_extra", 0.006291169673204422),
        ("taco_play", 0.03037598729133606 + 0.006291169673204422),
        ("toto", 0.19304856657981873),
        ("viola", 0.015149127691984177),
    ]
}

RTX_DOREMI_150K_MIX = [
    (dataset['path'].split('/')[0], dataset['weight']) for dataset in RTX_DOREMI_150K.values()
]

ROBOCASA_MIX = [
    ("robocasa", 1.0),
]

ROBOCASA_50_MIX = [
    ("robocasa_50", 1.0),
]

OXE_NAMED_MIXES = {
    "bridge": BRIDGE_MIX,
    "rtx": RT_X_MIX,
    "rtx_franka": RT_X_MIX + OXE_FRANKA_MIX,
    "oxe_magic_soup": OXE_MAGIC_SOUP,
    "oxe_full_soup": OXE_FULL_SOUP,
    "oxe_flex_act_soup": OXE_FLEX_ACT_SOUP,
    "google": GOOGLE_MIX,
    "google_bridge": GOOGLE_BRIDGE_MIX,
    "rtx_doremi_150k": RTX_DOREMI_150K_MIX,
    'oxe_diversity_soup': OXE_DIVERSITY_SOUP,
    'robocasa': ROBOCASA_MIX,
    'robocasa_50' : ROBOCASA_50_MIX,
}
