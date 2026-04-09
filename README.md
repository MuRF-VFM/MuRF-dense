# Multi-Resolution Fusion (MuRF)

This repo contains the code for the Multi-Resolution Fusion (MuRF) paper, which explores multi-resolution training for vision transformers across various dense prediction tasks.

## Environment

```bash
pip install torch torchvision torchmetrics transformers timm matplotlib scikit-image opencv-python pandas einops albumentationsx transforms
```

# Semantic Segmentation

This folder contains the semantic segmentation subexperiment for the Multi-Resolution Fusion paper

## Codebase Structure

* `train_mrf_ade20k.py`: Core MuRF training logic for ADE20K.
* `eval_mrf_ade20k.py`: Evaluation logic for ADE20K.
* `train_voc.py`: Baseline DINOv2 training loop for downstream segmentation (used for pipeline validation).
* `train_mrf_voc.py`: Core MuRF training logic.
* `test_voc.py`: Evaluation logic for the VOC splits.
* `train_cla_voc.py`: CLI and hyperparameter definitions.

## ADE20K

ADE20K is a large-scale semantic segmentation dataset with 150 classes, encompassing a wide variety of scenes and objects. 

### Data Preparation

ADE20K dataset can be downloaded from the [ADE20K website](https://ade20k.csail.mit.edu/) or [third-party repositories](https://huggingface.co/datasets/Gofinge/ADEChallengeData2016/tree/main)

The data structure should be as follows:
```
ADEChallengeData2016
├── images
│   ├── training
│   │   ├── ADE_train_00000001.jpg
│   │   ├── ...
│   │── validation
│   │   ├── ADE_val_00000001.jpg
│   │   ├── ...
├── annotations
│   ├── training
│   │   ├── ADE_train_00000001.png
│   │   ├── ...
│   ├── validation
│   │   ├── ADE_val_00000001.png
│   │   ├── ...
├── objectInfo150.txt
└── sceneCategories.txt
```

### Usage

To recreate baseline results (where we only use a single scale), one can run:

```bash
python3 semantic_segmentation/train_large_ade20k.py
```

The resolution can be set inside the python script `train_large_ade20k.py` using the variable `crop_size`

To recreate MuRF results, one can run:

```bash
python3 semantic_segmentation/train_mrf_ade20k.py
```

## Pascal VOC

Pascal VOC is a canonical benchmark dataset for semantic segmentation, featuring 20 foreground object classes and one background class. 

### Data Preparation

Consider the [mmseg data preparation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
markdown file for dataset preparation

The script at [semantic_segmentation/dataset/generate_voc.py](semantic_segmentation/dataset/generate_voc.py) might be useful for downloading the VOC dataset and converting it into a format compatible with our dataloader.

### Usage

To recreate baseline results (where we only use a single scale), one can run:
```bash
python -m semantic_segmentation.train_voc_cla --res 518
```
To recreate MuRF results, one can run:
```bash
python -m semantic_segmentation.train_voc_cla --res 140,266,518
```

# Depth Estimation

This folder contains the depth estimation subexperiment for the Multi-Resolution Fusion paper. Baseline functionality and configurations reproduce DINOv2 depth estimation, anchored on the official [DINOv2 ViT-B/14 NYU Linear Config](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_linear_config.py).

## Codebase Structure

### Architectures
* `dino_model.py`: Original DINOv2 baseline implementation.
* `mrf_model.py`: Multi-Resolution Fusion (MuRF) variant.

### Execution Pipelines
* `train_mrf.py`: Primary training loop for the MuRF architecture.
* `train_cla.py`: Command-line argument parsing and hyperparameter definitions.
* `train.py`: Baseline training loop (DINOv2 replication).
* `val.py`: Validation step execution and metric tracking.
* `test.py`: Final evaluation on designated test splits.

### Components
* **Losses**: Custom objective functions for depth regression.
* **Dataset**: Dataloaders and preprocessing pipelines for NYU Depth V2 and SUN RGB-D.
* **Callbacks**: Training hooks, including early stopping heuristics and qualitative depth map visualizations.
* **Extra (`ops.py`)**: Assorted tensor operations, including input unnormalization and ground truth depth map aesthetic formatting for NYU.

## NYU Depth V2

NYU Depth V2 is a widely used dataset for depth estimation, consisting of indoor scenes captured with RGB-D sensors. Following DINOv2's evaluation setup, we train on the NYU Depth V2 training set and evaluate on the NYU Depth V2 test set.

### Data Preparation

Download the dataset from the [Google Drive link](https://drive.google.com/file/d/1xI9ksHzCC_kUz6Z4FL_b1ttgj3RVHGwW/view?usp=sharing). The link can be found at DinoV3's [README](https://github.com/facebookresearch/dinov3/blob/main/DATASETS.md). Unzip the downloaded file and ensure the data structure is as follows:
```
nyu
├── basement_0001a
├── basement_0001b
...
├── nyu_test.txt
└── nyu_train.txt
```

### Usage
Please refer to the "SUN RGB-D" section below for training and evaluation instructions, as the both the results for NYU Depth V2 and SUN RGB-D will be reported there.

## SUN RGB-D

Following DinoV2's evaluation setup, we evaluate the model trained on NYU Depth V2 directly on the SUN RGB-D test set without any fine-tuning.

The result will be printed after you run the training script mentioned in the NYU section.

### Data Preparation

Download the dataset from the [SUN RGB-D website](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip). The dataset includes RGB images, depth maps, and corresponding annotations. After downloading, ensure the data structure is as follows:

```
SUNRGBD
├── kv1
├── kv2
├── realsense
└── xtion
```

Download the dataset split file from https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/splits/SUNRGBD_val_splits.txt , and put it into `depth_estimation/SUNRGBD_val_splits.txt`

### Usage
To recreate baseline results (where we only use a single scale), one can run:
```bash
python -m depth_estimation.train_cla --scales 1.0 --model-size base --lin lin1 --max-iters 38400
```
To recreate MuRF results, one can run:
```bash
python -m depth_estimation.train_cla --scales 0.5,1.0,1.5 --model-size base --lin lin1 --max-iters 38400
```
