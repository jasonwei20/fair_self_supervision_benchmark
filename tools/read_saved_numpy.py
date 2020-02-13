import numpy as np

# input_features_file = "resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_features.npy"
# input_inds_file = "resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_inds.npy"
# input_targets_file = "resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_targets.npy"

# features = np.load(input_features_file)
# inds = np.load(input_inds_file)
# targets = np.load(input_targets_file)

x = np.load("resnet_outputs_test/trainval_res3_3_branch2c_bn_s0_s5k13_resize_features.npy")
print(x.shape)
# print(x[:5, :5])

# print(features.shape)
# print(inds.shape)
# print(targets)