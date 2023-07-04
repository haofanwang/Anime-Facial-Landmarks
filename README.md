# Anime Facial Landmarks


## Installation

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmdet==2.25.1
pip install mmpose==0.28.1

git clone https://github.com/haofanwang/Anime-Facial-Landmarks.git
```

This package is tested only on Ubuntu.

```python
import cv2

from anime_face_detector import create_detector

detector = create_detector('yolov3')
image = cv2.imread('assets/input.jpg')
preds = detector(image)
print(preds[0])
```

```
{'bbox': array([2.2450244e+03, 1.5940223e+03, 2.4116030e+03, 1.7458063e+03,
        9.9987185e-01], dtype=float32),
 'keypoints': array([[2.2593938e+03, 1.6680436e+03, 9.3236601e-01],
        [2.2825300e+03, 1.7051841e+03, 8.7208068e-01],
        [2.3412151e+03, 1.7281011e+03, 1.0052248e+00],
        [2.3941377e+03, 1.6825046e+03, 5.9705663e-01],
        [2.4039426e+03, 1.6541921e+03, 8.7139702e-01],
        [2.2625220e+03, 1.6330233e+03, 9.7608268e-01],
        [2.2804077e+03, 1.6408495e+03, 1.0021354e+00],
        [2.2969380e+03, 1.6494972e+03, 9.7812974e-01],
        [2.3357908e+03, 1.6453258e+03, 9.8418534e-01],
        [2.3475276e+03, 1.6355408e+03, 9.5060223e-01],
        [2.3612463e+03, 1.6262626e+03, 9.0553057e-01],
        [2.2682278e+03, 1.6631940e+03, 9.5465249e-01],
        [2.2814783e+03, 1.6616484e+03, 9.0782022e-01],
        [2.2987590e+03, 1.6692812e+03, 9.0256405e-01],
        [2.2833625e+03, 1.6879142e+03, 8.0303693e-01],
        [2.2934949e+03, 1.6909009e+03, 8.9718056e-01],
        [2.3021218e+03, 1.6863715e+03, 9.3882143e-01],
        [2.3471826e+03, 1.6636573e+03, 9.5727938e-01],
        [2.3677822e+03, 1.6540554e+03, 9.4890594e-01],
        [2.3889211e+03, 1.6611255e+03, 9.5125675e-01],
        [2.3575544e+03, 1.6800433e+03, 8.5919142e-01],
        [2.3688926e+03, 1.6800665e+03, 8.3275074e-01],
        [2.3804905e+03, 1.6761322e+03, 8.4160626e-01],
        [2.3165366e+03, 1.6947096e+03, 9.1840971e-01],
        [2.3282458e+03, 1.7104808e+03, 8.8045174e-01],
        [2.3380054e+03, 1.7114034e+03, 8.8357794e-01],
        [2.3485500e+03, 1.7080273e+03, 8.6284375e-01],
        [2.3378748e+03, 1.7118135e+03, 9.7880816e-01]], dtype=float32)}
```

### Pretrained models

[Here](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1) are the pretrained models.
(They will be automatically downloaded when you use them.)
