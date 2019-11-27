# Thru-Restaurant

## 1. Features

- Vehicle total time in system from start to finish
- Total vehicles by time of day
- Wait times from point to point
- Number of vehicles leaving line before ordering (after what period of time)
- Access to video feeds live
- Wait times at relevant locations (Order points, cash booth, present booth, wait parking spot)
- Ability to record multiple feeds and playback at variable speeds for rapid video review

#### 1.1. Object Detector
The vechicle detector was based on `YOLOv3` from [here](https://pjreddie.com/darknet/yolo/), and implemented using 
- `OpneCV`'s dnn
- [Darknet](https://pjreddie.com/darknet/) based on `Tensorflow` (with `keras` background)

#### 1.2. Object Tracker
The used tracking algorithms are 
- `Dlib`'s `cross-correlate` tracking algorithms
- `OpenCV`'s `CRST` and `MOSSE`.



## 2. Installation

- Download the repository
    
        git clone https://gitlab.com/drimyus/thru-data.git

- Install the dependencies

        bash setup.sh


## 3. Settings

#### 3.1 Video Feed Settings
fps: 20
width: 1920.0
height: 1080.0


#### 3.1. Other setting parameters:
Open the `settings.py` and change the parameters 

    nano settings.py
    
- Trackers ( line 24 on `settings.py`):
  
        # [Tracker]
        TRACKER = "MOSSE"

   Available trackers are `DLIB`, `MOSSE` and `CSRT`, the default is `DLIB`
   
   
- Detectors (line 27 on `settings.py`):         
       
        # [Detector]
        DETECTOR = "YOLO3"
    
    Available detectors are `DARKNET3` and `YOLO3`, the default is `YOLO3`  


#### 3.2. db for user registration (optional)
- create user db 
        
        python3 endpoint.py db init

- create db migrations        
        
        python3 endpoint.py db migrate
    
- upgrade db using created migrations
    
        python3 endpoint.py db upgrade
    

## 4. Running

- start recording

    python3 saver.py


- start analyzing
    
    python3 main.py
