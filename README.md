# Simple and Lightwight Human Pose Estimation

## Introduction
The reproduced paper [*Simple and Lightwight Human Pose Estimation and Tracking*](https://arxiv.org/abs/1911.10346).On COCO keypoints valid dataset, if with_gcb module  achieves **66.5 of mAP**, else **64.4 of mAp** </br>

## Main Results
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |

 Arch | with_GCB | AP    | Ap .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) 
 - | :-: | :-: | :-::-: | :-: | :-: | :-: | :-: | :-: | :-: | -: 
 256x192_lp_net_50_d256d256 |**yes** | 0.665 | 0.903 | 0.746 | 0.644 | 0.697 | 0.700 | 0.911 | 0.771 | 0.672 | 0.743 |
 256x192_lp_net_50_d256d256 |**no** | 0.644 | 0.885 | 0.715 | 0.619 | 0.685 | 0.679 | 0.898 | 0.742 | 0.647 | 0.725 |
### Note:
- Flip test is used.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Disable cudnn for batch_norm:
   ```
   # PYTORCH=/path/to/pytorch
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed  and then set an environment variable (PYTORCH in this case) accordingly.
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
7. coco pretrained models under ${POSE_ROOT}/models/pytorch, and it looks like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- lp_coco
                |-- lp_net_50_256x192_with_gcb.pth.tar
                `-- lp_net_50_256x192_without_gcb.pth.tar
   ```

4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── images
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation
**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
### Valid on COCO val2017 using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/lp_net50/256x192_d256x2_adam_lr1e-3_lp.yaml \
    --flip-test \
    --model-file models/lp_coco/lp_net_50_256x192_with_gcb.pth.tar
```

### Training on COCO train2017

```
python pose_estimation/train.py \
    --cfg experiments/coco/lp_net50/256x192_d256x2_adam_lr1e-3_lp.yaml
```
### Demo
The region of human need to be given in ```demo.py```

```
python pose_estimation/demo.py \
    --cfg experiments/coco/lp_net50/256x192_d256x2_adam_lr1e-3_lp.yaml \
    --model-file ./models/lp_coco/lp_net_50_256x192_with_gcb.pth.tar
    --img-file ./images/0.jpg
```
