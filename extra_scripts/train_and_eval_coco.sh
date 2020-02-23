mkdir voc_extracted_features/$1 &&
python tools/extract_features.py \
    --config_file configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir voc_extracted_features/$1 \
    TEST.PARAMS_FILE converted_models/$1.pkl \
    TRAIN.DATA_FILE voc_data/train_images.npy \
    TRAIN.LABELS_FILE voc_data/train_labels.npy &&
python tools/extract_features.py \
    --config_file configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml \
    --data_type test \
    --output_file_prefix test \
    --output_dir voc_extracted_features/$1 \
    TEST.PARAMS_FILE converted_models/$1.pkl \
    TEST.DATA_FILE voc_data/test_images.npy \
    TEST.LABELS_FILE voc_data/test_labels.npy &&
python tools/svm/train_svm_kfold.py \
    --data_file voc_extracted_features/$1/trainval_res5_2_branch2c_bn_s0_s1k6_resize_features.npy \
    --targets_data_file voc_extracted_features/$1/trainval_res5_2_branch2c_bn_s0_s1k6_resize_targets.npy \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path voc_svm/conv5/$1/ &&
python tools/svm/test_svm.py \
    --data_file voc_extracted_features/$1/test_res5_2_branch2c_bn_s0_s1k6_resize_features.npy \
    --targets_data_file voc_extracted_features/$1/test_res5_2_branch2c_bn_s0_s1k6_resize_targets.npy \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path voc_svm/conv5/$1/

# mkdir voc_extracted_features/resnet50_default_random &&
# python tools/extract_features.py \
#     --config_file configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml \
#     --data_type train \
#     --output_file_prefix trainval \
#     --output_dir voc_extracted_features/resnet50_default_random \
#     TEST.PARAMS_FILE converted_models/resnet50_default_random.pkl \
#     TRAIN.DATA_FILE voc_data/train_images.npy \
#     TRAIN.LABELS_FILE voc_data/train_labels.npy &&
# python tools/extract_features.py \
#     --config_file configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml \
#     --data_type test \
#     --output_file_prefix test \
#     --output_dir voc_extracted_features/resnet50_default_random \
#     TEST.PARAMS_FILE converted_models/resnet50_default_random.pkl \
#     TEST.DATA_FILE voc_data/test_images.npy \
#     TEST.LABELS_FILE voc_data/test_labels.npy &&
# python tools/svm/train_svm_kfold.py \
#     --data_file voc_extracted_features/resnet50_default_random/trainval_res5_2_branch2c_bn_s0_s1k6_resize_features.npy \
#     --targets_data_file voc_extracted_features/resnet50_default_random/trainval_res5_2_branch2c_bn_s0_s1k6_resize_targets.npy \
#     --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
#     --output_path voc_svm/conv5/resnet50_default_random/ &&
# python tools/svm/test_svm.py \
#     --data_file voc_extracted_features/resnet50_default_random/test_res5_2_branch2c_bn_s0_s1k6_resize_features.npy \
#     --targets_data_file voc_extracted_features/resnet50_default_random/test_res5_2_branch2c_bn_s0_s1k6_resize_targets.npy \
#   --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
#     --output_path voc_svm/conv5/resnet50_default_random/ 