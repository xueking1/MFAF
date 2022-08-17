# Multi-source Feature Adaptive Fusion for 3D Object Detection

MFAF is a two-stage 3D object detection method.Meanwhile.It is efficiency and accuracy stereo 3D object detection method for autonomous driving.

## Introduction

a multi-source feature fusion method based on adaptive fusion strategy is designed in this paper, which is called MFAF (Multi-source Feature Adaptive Fusion). Multi-source features refer to image semantic features, voxel features and point features, and try to exploit the advantages of three data formats in one fusion algorithm at the same time.The whole network consists of three main modules: (1)Image branch ,(2) LiDAR branch , and (3) Voxel branch. Given an original feature map, the object point probability estimation and multi-scale feature map are obtained by image feature encoder. Then, certain point clouds are sampled to get the set of key points, and voxel features and 3D bird's eye view feature map are obtained by voxel branch. Finally, the multi-scale feature map and voxel features are output to the LiDAR branch, which is fused with the key point set to obtain the multi-feature fused key points. The global features are obtained by combining the image, LiDAR and voxel branch to predict the detection results.

## Requirements

- Linux (tested on Ubuntu 18.04)
- Python 3.7+
- PyTorch 1.6
- mmdet 2.25.1
- mmdet3d 1.0.0rc4
- mmcv 1.6.1
- mmsegmentation 0.27.0

## Installation

Refer to the following four sections to install MFAF.

### a.mmdet

Refer to  [GitHub - MMDetection](https://github.com/open-mmlab/mmdetection) to install mmdet.

### b.mmcv

Refer to [GitHub - MMCV](https://github.com/open-mmlab/mmcv) to install mmcv.

### c.mmsegmentation

Refer to [GitHub -mmsegmentation ](https://github.com/open-mmlab/mmsegmentation) to install mmsegmentation.

### d.mmdet3d

Refer to [GitHub - MmDetection3D](https://github.com/open-mmlab/mmdetection3d) to install mmdet3d.

## File Structure

Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
MFAF
├─configs
│  ├─MFAF    <-- MV3D net related source code 
│  │      mfaf_kitti-3d-3class.py
├─data     <-- all data is stored here.
│  ├─kitti
│  │  │  kitti_dbinfos_train.pkl
│  │  │  kitti_infos_test.pkl
│  │  │  kitti_infos_train.pkl
│  │  │  kitti_infos_trainval.pkl
│  │  │  kitti_infos_val.pkl
│  │  │  Readme.txt
│  │  ├─.ipynb_checkpoints
│  │  ├─gt_database
│  │  ├─ImageSets
│  │  ├─testing
│  │  │  ├─calib & velodyne & image_2
│  │  ├──training
│  │  │  ├──calib & velodyne & label_2 & image_2 & (optional: planes)
├─demo
├─docker
├─docs
├─mmdet3d
├─requirements
├─resources
├─tests
├─tools
    │  create_data.py
    │  create_data.sh
    │  dist_test.sh
    │  dist_train.sh
    │  slurm_test.sh
    │  slurm_train.sh
    │  test.py
    │  train.py    <--- training the whole network. 
    │  update_data_coords.py
    │  update_data_coords.sh
    ├─analysis_tools
    ├─data_converter
    ├─deployment
    ├─misc
    └─model_converters
│  README.md
│  requirements.txt
│  setup.cfg
│  setup.py
```

## Modification needed to run

Follow Installation. After installing the environment, perform the following steps:

a.This step generates the KITTI_gT_database and.pkl. json files shown below.

```python
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

b.Training the KITTI dataset.

First you should set the data path in mfaf_kitti-3D-3class.py.

```python
data_root = '/root/data/kitti/'
```

You can then train by running the following code

```python
python ./tools/train.py ./configs/MFAF/mfaf_kitti-3d-3class.py
```

c.Testing

```
 python ./tools/test.py ./configs/MFAF/mfaf_kitti-3d-3class.py /root/mmdetection3d-master/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth  --eval mAP
```

As shown in mfaf_kitti-3D-3class.py, you need to download PTH in advance

```python
load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'  # noqa
```

## 

## Some other readme.md files inside this repo

- README-mmdet3d.md:How to install MMDET3D
