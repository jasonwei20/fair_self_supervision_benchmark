import numpy as np

# input_features_file = "resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_features.npy"
# input_inds_file = "resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_inds.npy"
# input_targets_file = "resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_targets.npy"

# features = np.load(input_features_file)
# inds = np.load(input_inds_file)
# targets = np.load(input_targets_file)

# x = np.load("/home/brenta/scratch/jason/outputs/voc/example_features/conv5/resnet_50_in_pretrained.npy")
x = np.load("/home/brenta/scratch/jason/outputs/voc/example_features/conv5/resnet_50_in_random_uniform_one_file_big.npy")
# x = np.load("/home/brenta/scratch/jason/outputs/voc/example_features/resnet_50_in_random.npy")
# x = np.load("/home/brenta/scratch/jason/outputs/voc/example_features/resnet_50_e0_mb_10000_va0.51130.npy")
# x = np.load("random_numpy_array.npy")
# x = np.load("../fair_self_supervision_benchmark/voc_data/train_images.npy")
print(x.shape)

for i in range(50):
    print_line = f"{x[0][i]:.3f}\t{x[1][i]:.3f}\t{x[2][i]:.3f}\t{x[2][i]:.4f}\t{x[2][i]:.5f}"
    print(print_line)
    # print(x[0][i], x[1][i], x[2][i])
# print(x[-5:])

# x = np.random.rand(2501, 9216)
# np.save("random_numpy_array.npy", x)

# print(features.shape)
# print(inds.shape)
# print(targets)