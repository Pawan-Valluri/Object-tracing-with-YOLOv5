# Object-tracing-with-YOLOv5

<div align="center">
<p>
<img src="road01.gif" width="633" height="360"/> 
</p>
<br>
<div>

</div>

</div>

## Introduction 
This repository contains multiple scripts for various tasks. Each script preforms a separate task. The detections are generated by [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset, are passed to a set of algorithms that track/count/monitor the detected objects. The scripts can perform these operations with any set of yolov5 pretrained weights.

I'll update this repository with more scripts when I am able. 

## Installation:
1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/Pawan-Valluri/Object-tracing-with-YOLOv5.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including torch==1.9.0 (with cuda 10.2 or 11.3 for better peformance). To install, run:

`pip install -r requirements.txt`

## Usage

These scripts can be run on most video formats

```bash
$ python path_counter_per_area_per_class.py --source 0  # webcam
                                                     img.jpg  # image
                                                     vid.mp4  # video
                                                     path/  # directory
                                                     path/*.jpg  # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

## Using pretrained weights

These scripts can work an any yolov5 pretrained weights. You can use your weights by executing the following commands.

```bash


$ python path_tracer.py --source 0 --yolo_model yolov5n.pt --img 640
                                                yolov5s.pt
                                                yolov5m.pt
                                                yolov5l.pt 
                                                yolov5x.pt --img 1280
                                                customyolo.pt
                                            ...
```


