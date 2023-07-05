import cv2
import torch
from detector import LandmarkDetector


if __name__ == "__main__":
    
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

        x1, y1, x2, y2, conf_bbox = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (j, (x, y,conf_kpt)) in enumerate(kpts):
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            image = cv2.putText(image, str(j), (x,y), fontFace=0, fontScale=0.6, color=(255, 0, 0), thickness=1)
    
    cv2.imwrite("assets/output.jpg", image)