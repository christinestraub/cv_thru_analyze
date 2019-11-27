## Project Structure

    thru-data
    |
    ├── config			# configuration
    │   ├── infos           # information about the camera, lens distortion, section definition, etc (json format)
    |   ├── logs            # logs for running of main.py
    |   └── models          # CNN models and predefined (COCO and PASACL) label information
    │       ├── label_names
    │       ├── ssd
    │       └── yolo
    |
    ├── data           	# workspace directory for excutable
    │   ├── images         	# sample images to test
    │   │	└── *.jpg
    │   │
    │   └── videos          # saved videos by saver.py
    |
    │
    ├── src                    	# src
    │   ├── analyze	          		
    │   │	├── roi_utils.py        # utilities of roi events(e.g. crossing line, get in area, escap area, etc) 
    │   │   ├── set_roi.py          # definition the roi using mouse events on simple UI(movements and clicking)
    │   │   └── roi.json            # defined roi (json format)
    │   │
    │   ├── cv         		    # main computer vision modules 
    │   │	├── calib               # calibration utilities
    │   │   ├── detect              # utilities for detecting the object (only street objects vehicle)
    │   │   ├── draw                # utilities for drawing detected/tracking objects, rois
    │   │   └── track               # tracking utilities
    │   │
    │   ├── static              # web browser implementation
    │   │
    │   └── templates         	# web browser implementation
    |
    ├── utils
    │   ├── common.py           # common functions
    │   ├── constant.py         # constants through the project
    │   └── logger.py           # utilities for logging
    |
    ├── endpoint.py             # endpoint of analyzing module 
    │
    ├── main.py (mainthread)    # analyzing the saved videos
    |
    ├── saver.py (mainthread)   # saving the stream as a local file
    |
    ├── settings.py             # user interface for setting params 
    |
    └── setup.sh                # bash script to install the dependencies


## Settings

#### Settings for CNN model

There are several available CNN models, they are selected on `settings.py`. The keywords for each CNN models are:  

| Models           | Keyword          |
| :--------------: | :--------------: |
| SSD Mobile       | SSD              |
| SSD VGG          | VGG              |
| Yolo v2 OpenCV   | YOLO2            |
| Yolo v2 darknet  | DARKNET2         |
| Yolo v3 OpenCV   (default) | YOLO3            |
| Yolo V3 darknet  | DARKNET3         |

#### Settings for Tracking

There are several available tracking APIs, they are selected on `settings.py`. The keywords for each APIs are:

| Tracker          | Keyword          |
| :--------------: | :--------------: |
| Cross-Correlate tracker DLib  (default)    | DLIB              |
| CSRT tracker OpenCV          | CSRT              |
| MOSSE tracker OpenCV   | MOSSE            |


#### Other Params

`B_SAVE_VIDEO` is `True` then the result video will be saved.

`B_SHOW_VIDEO` is `True` then showing the frames with the detected and tracking objects together.

`VIDEO_SRC` feed definition of video stream. e.g. "rtsp://admin:streetcam123@47.40.117.222:8889/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
 

## ROIs

For analyzing the vehicle thru, `tracking-area`, 2 x `order-point`, a `cross-line` were defined. Here is showing the set rois.

![ROIs](rois.png)

## Export Result

The analyzing result was saved as csv format which including the timestamp logging history per each vehicle.
```
ID	 AppearanceTime (sec)	 TimeToOrderPoint1 (sec)	 TimeAtOrderPoint1 (sec)	 Exist Vehicle in Front (T/F)	 TimeToOrderPoint2 (sec)	 TimeAtOrderPoint2 (sec) 
4	 2018-12-28 20:15:02	    56.8	                    3.6 	                    False	                        67.2	                    23.9
5	 2018-12-28 20:15:02	    66.5	                    20.8	                    True	                        97.2                        0
7	 2018-12-28 20:15:26	    74.3	                    21.9	                    True	                        138.7	                    3.6
8	 2018-12-28 20:15:58	    93.3	                    47  	                    True	                        143.8	                    1.6
9	 2018-12-28 20:16:16	    128.7	                    5   	                    False	                        138.7	                    56.2
10	 2018-12-28 20:17:12	    84.5	                    13.8	                    True	                        134.1	                    9.6
``
