
# Face Mask Detection with Darknet53 　　　　　　　　
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><a href="https://colab.research.google.com/github/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/darknet53_mask_origin.ipynb" target="_parent\">![image](https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/demo_video/clip.png)

### *Feature*         

Model: Easily try Darknet53.conv.74

Environment: Google colab 

Evaluate: mAP@0.50 = 88.78 %

Classes: mask / no mask / mask improperly

Demo:  [Demo on Youtube](https://youtu.be/H3QA6NHCdqw)

## Dataset
The dataset for this pretrained network is provided by [VictorLin000](https://github.com/VictorLin000/YOLOv3_mask_detect) and contains 678 images of people with and without masks. There are 3 different classes annotated:

* `no mask` - No mask at all.
* `improperly` - Partially covered face.
* `mask` - Mask covers the essential parts.

You can download the dataset directly from [google drive](https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG).

## Model structure
### Darknet-53
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/model.png">

Darknet-53 network uses successive 3 × 3 and 1 × 1 convolutional layers but now has some shortcut connections as well and is significantly larger. It has 53 convolutional layers.
YOLO v2 originally used a custom deep architecture darknet-19 which has 19 layers with 11 additional layers for object detection. YOLO v2 often suffers from small object detection. This is because the layers downsampled the input. To solve this problem, YOLO v2 used an identity mapping to link the feature maps of the previous layer to capture lower-level features (which is able to reflect small object feature).
However, the architecture of YOLO v2 still lacked the most important elements required by detecting small object. YOLO v3 puts it all together.
First, YOLO v3 uses a variant of Darknet with 53 layer networks trained on "Imagnet".

### Why 106 layers?
![image](https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/yolov3.png)
 For detection tasks, 53 more layers are stacked, so YOLO v3 has a 106 layer fully convolution. That's why YOLO v3 is slower than YOLO v2. You can see this architecture of YOLO v3 in colab. ( If you want more information about YOLO v3 & image detection, check powerpoint file in this repo)
 
```
!./darknet detector map /content/Face_Mask_Detection_YOLO/MASK/object.data\
                        /content/Face_Mask_Detection_YOLO/MASK/detect_mask.cfg\
                        /content/darknet/backup
 ```

## Demo on three cases
### mask improperly
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/mask_incorrect/detected.jpg" width="60%">

### mask
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/with_mask/detected.jpg" width="60%">

### no mask
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/without_mask/detected.jpg" width="60%">

## Demo on special cases
### cloth_mask
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/special_mask/cloth_mask/detected.jpg" width="60%">

### n95_n99
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/special_mask/n95_n99/detected.jpg" width="60%">

### scarf
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/special_mask/scarf/detected.jpg" width="60%">

### transparent
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/special_mask/transparent/detected.jpg" width="60%">


## Evaluation
### Learning Curve
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_origin/%E5%8E%9F%E5%A7%8B%E6%95%B8%E6%93%9A%E6%88%AA%E5%9C%96/learning_curve.png" width="60%">

### Average Precision
| class_id | name       | TP  | FP | ap     |
|----------|------------|-----|----|--------|
| 0        | mask       | 312 | 34 | 94.50% |
| 1        | improperly | 17  | 0  | 96.77% |
| 2        | no mask    | 53  | 10 | 75.07% |

### F1-score & Average IoU
| conf_thresh | precision | recall | F1-score | TP  | FP | FN | average IoU |
|-------------|-----------|--------|----------|-----|----|----|-------------|
| **0.25**        |    0.90   |  0.91  |   0.90   | 382 | 44 | 39 |   72.08 %   |

### Test Result
| class_id | name       | TP  | FP | ap     |
|----------|------------|-----|----|--------|
| 0        | mask       |  25 |  2 | 92.59% |
| 1        | improperly |  25 |  8 | 75.76% |
| 2        | no mask    |  24 |  2 | 92.31% |

### Mean Average Precision
mean average precision (mAP@0.50) = 88.78 %

## Usage
### Load weight
```
!wget https://pjreddie.com/media/files/darknet53.conv.74
```

or you can get pretrained [weights](https://drive.google.com/drive/folders/16MsdDvPuF6CxFd0vW2VYya6e5cqZJjZI?usp=sharing)
for this data
### Train
```
!./darknet detector train /content/Face_Mask_Detection_YOLO/Mask/object.data\
                          /content/Face_Mask_Detection_YOLO/Mask/detect_mask.cfg\
                          darknet53.conv.74\
                          -dont_show -map 
```
### Detect
```
!./darknet detector test /content/Face_Mask_Detection_YOLO/Mask/object.data\
                         /content/Face_Mask_Detection_YOLO/Mask/detect_mask.cfg\
                         /content/backup/detect_mask_last.weights\
                         /content/Face_Mask_Detection_YOLO/demo/man_0_1.png
```

## Addition (with Large Dataset)
I've tried large dataset(FFHQ + MFN) on darknet53, but it seems overfit on the dataset and output all the bounding boxes with whole image.

## Evaluation
### Learning Curve
<img src="https://github.com/LinGeorge/Face_Mask_Detection_Darknet/blob/master/darknet53_mask_MFN/%E5%8E%9F%E5%A7%8B%E6%95%B8%E6%93%9A%E6%88%AA%E5%9C%96/learning_curve.png" width="60%">

### Average Precision
| class_id | name       | TP  | FP | ap     |
|----------|------------|-----|----|--------|
| 0        | mask       | 104 |  0 | 100.00%|
| 1        | improperly | 17  |  1 | 100.00%|
| 2        | no mask    | 53  |  1 | 100.00%|

### F1-score & Average IoU
| conf_thresh | precision | recall | F1-score | TP  | FP | FN | average IoU |
|-------------|-----------|--------|----------|-----|----|----|-------------|
| **0.25**        |    0.99   |  1.00  |   1.00   | 300 | 2 | 0 |   90.03 %   |

### Test Result ( the reason why the model overfits on the dataset distribution )
| class_id | name       | TP  | FP | ap     |
|----------|------------|-----|----|--------|
| 0        | mask       |  24 |  5 | 82.76% |
| 1        | improperly |   2 | 31 |  6.06% |
| 2        | no mask    |  25 |  7 | 78.13% |

### Mean Average Precision
mean average precision (mAP@0.50) = 100.00 %
 
## Reference
- https://arxiv.org/abs/1804.02767
- https://arxiv.org/abs/1506.02640 
- https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
- https://github.com/AlexeyAB/darknet
- https://github.com/VictorLin000/YOLOv3_mask_detect
- https://pjreddie.com/darknet/yolo/
- https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg
- https://medium.com/@artinte7
- https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
- https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/