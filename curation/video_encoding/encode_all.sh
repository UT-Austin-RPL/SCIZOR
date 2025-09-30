DATA_DIR=<path_to_data_dir>
OUTPUT_DIR=<path_to_output_dir>
CACHE_DIR=<path_to_store_model>
BATCH_SIZE=256

# format: [dataset_name, dataset_version, transforms]
DATASET_TRANSFORMS=(
    "taco_play 0.1.0 resize_and_jpeg_encode"
    "jaco_play 0.1.0 resize_and_jpeg_encode"
    "berkeley_cable_routing 0.1.0 resize_and_jpeg_encode"
    "roboturk 0.1.0 resize_and_jpeg_encode"
    "nyu_door_opening_surprising_effectiveness 0.1.0 resize_and_jpeg_encode"
    "viola 0.1.0 resize_and_jpeg_encode"
    "berkeley_autolab_ur5 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels"
    "toto 0.1.0 resize_and_jpeg_encode"
    "language_table 0.1.0 resize_and_jpeg_encode"
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "bc_z 0.1.0 resize_and_jpeg_encode"
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
    "utaustin_mutex 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
    "berkeley_fanuc_manipulation 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
    "cmu_stretch 0.1.0 resize_and_jpeg_encode"
    "fractal20220817_data 0.1.0 resize_and_jpeg_encode"
    "bridge_dataset 1.0.0 resize_and_jpeg_encode"
    "kuka 0.1.0 resize_and_jpeg_encode,filter_success"
)

for tuple in "${DATASET_TRANSFORMS[@]}"; do
  # Extract strings from the tuple
  strings=($tuple)
  DATASET=${strings[0]}
  python -m curation.video_encoding.video_encode_cosmos --dataset_name $DATASET --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --cache_dir $CACHE_DIR
done
