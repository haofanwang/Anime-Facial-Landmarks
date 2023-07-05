# Anime Facial Landmarks
This repo is a lightweight migration of [anime-face-detector](https://github.com/hysts/anime-face-detector) for anime facial landmarks detection.

<right><img src="https://github.com/haofanwang/Anime-Facial-Landmarks/raw/main/assets//output.jpg" width="49%" height="49%"></right> 

## Installation
As the codebase is highly based on a specific version of mmcv, mmdet and mmpose. To avoid confilct, it is recommendated to follow our instructions below. If anyone want to migrate
it to lastest version, please make a pull requests.

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmdet==2.25.1
pip install mmpose==0.28.1
pip install opencv-python-headless>=4.5.4.58
```

### Pretrained models

The pretrained models can be downloaded from [Here](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1). Please put them under `./checkpoints`.


## Usage

```python
import cv2
import torch
from detector import LandmarkDetector

# configs
detector_config_path = "./configs/mmdet/yolov3.py"
landmark_config_path = "./configs/mmpose/hrnetv2.py"

# ckpts
detector_checkpoint_path = "./checkpoints/mmdet_anime-face_yolov3.pth"
landmark_checkpoint_path = "./checkpoints/mmpose_anime-face_hrnetv2.pth"

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init model
detector = LandmarkDetector(landmark_config_path,
                        landmark_checkpoint_path,
                        detector_config_path,
                        detector_checkpoint_path,
                        device=device)

# inference
image = cv2.imread('assets/input.jpg')
preds = detector(image)

# loop over the detections
for (i, pred) in enumerate(preds):
        bbox = pred["bbox"]
        kpts = pred["keypoints"]
```
The order of keypoints can found in the sample Shinichi Kudo image above.


