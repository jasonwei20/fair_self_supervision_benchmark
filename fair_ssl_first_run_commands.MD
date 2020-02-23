```
(ssl-benchmark) checkpoints$ ls /home/brenta/scratch/jason/checkpoints/example_models/
resnet50_default_random.pt        resnet50_e0_mb30000_va0.68785.pt  resnet50_e1_mb100000_va0.76889.pt
resnet50_e0_mb10000_va0.51130.pt  resnet50_e0_mb40000_va0.71450.pt  resnet50_e2_mb210000_va0.79200.pt
resnet50_e0_mb20000_va0.65573.pt  resnet50_e0_mb60000_va0.74241.pt  resnet50_in_pretrained.pt
```

```
python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_default_random.pt \
	--output_model=converted_models/resnet50_default_random.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e0_mb10000_va0.51130.pt \
	--output_model=converted_models/resnet50_e0_mb10000_va0.51130.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e0_mb20000_va0.65573.pt \
	--output_model=converted_models/resnet50_e0_mb20000_va0.65573.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e0_mb30000_va0.68785.pt \
	--output_model=converted_models/resnet50_e0_mb30000_va0.68785.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e0_mb40000_va0.71450.pt \
	--output_model=converted_models/resnet50_e0_mb40000_va0.71450.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e0_mb60000_va0.74241.pt \
	--output_model=converted_models/resnet50_e0_mb60000_va0.74241.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e1_mb100000_va0.76889.pt \
	--output_model=converted_models/resnet50_e1_mb100000_va0.76889.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_e2_mb210000_va0.79200.pt \
	--output_model=converted_models/resnet50_e2_mb210000_va0.79200.pkl


python extra_scripts/pickle_pytorch_to_caffe2.py \
	--pth_model=/home/brenta/scratch/jason/checkpoints/example_models/resnet50_in_pretrained.pt \
	--output_model=converted_models/resnet50_in_pretrained.pkl
```

```
./extra_scripts/train_and_eval_coco.sh resnet50_in_pretrained
./extra_scripts/train_and_eval_coco.sh resnet50_e0_mb10000_va0.51130
./extra_scripts/train_and_eval_coco.sh resnet50_e0_mb20000_va0.65573
./extra_scripts/train_and_eval_coco.sh resnet50_e0_mb30000_va0.68785
./extra_scripts/train_and_eval_coco.sh resnet50_e0_mb40000_va0.71450


./extra_scripts/train_and_eval_coco.sh resnet50_e0_mb60000_va0.74241
./extra_scripts/train_and_eval_coco.sh resnet50_e1_mb100000_va0.76889
./extra_scripts/train_and_eval_coco.sh resnet50_e2_mb210000_va0.79200
```
