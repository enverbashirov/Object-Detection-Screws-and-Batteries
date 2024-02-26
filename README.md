# A *Computer Vision* case study: Screws and Batteries
##### Object Detection on the given images
<img src="https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/blob/master/results/combined/screws.jpg?raw=true" width="480"/> <img src="https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/blob/master/results/combined/batteries.jpg?raw=true" width="480"/>

#### Coding Challenge: Simulating Computer Vision for Autonomous Battery Disassembly
The task required here was to detect the **screws** and **batteries** in the given two sample images. For the simplicity, first image with the container having a closed lid and the second image with the batteries displayed are called `screws.jpg` and `batteries.png` respectively. The *Object Detection* using Neural Network approach has been chosen for this specific task. Furthermore, *YOLOv8* has been used to perform the detection. Two separate models have been trained with two different datasets for the purpose of identifying screws and batteries. Thus, the two objects were detected separately.

See [the Challenges and What Else? section](#challenges-and-hat-else?) for details on what issues / constraints the current scenario had
See [the Final Takes section](#final-takes) for what is my take on what could be done to improve this detection task and also explanation of pose estimation and how it could be tackles.

##### | -------------------  Screws.jpg  ---------------------- | --------------------  Battery.png  -------------------- |
<img src="https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/blob/master/datasets/screws.jpg?raw=true" width="320"/> <img src="https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/blob/master/datasets/battery.png?raw=true" width="320"/>

### Challenges and What Else?
- Constraint: Quality of the photos
	- Images in the datasets were not bad for generalizing but for a specific scenario such as the one provided in the `screws.jpg` and `battery.png` a tailor made set of images would be better to use. 

- Pre-processing before using a DL approach
	- Pre-processed image with some of the well known filters/edge detectors/contour detectors may give a decent result. This needs to be tested and evaluated


### Technical Details
File Structure
```
├───datasets
│   ├───batteries
│   │   ├───test
│   │   ├───train
│   │   └───valid
│   └───screws
│       ├───test
│       ├───train
│       └───valid
├───models
├───results
│   ├───batteries (detection of only batteries)
│   ├───combined 	(detection of both batteries and screws)
│   └───screws 		(detection of only screws)
├───runs
│   └───detect
│       ├───batteries	(figures/graphs for the performance of the batteries model)
│       └───screws		(figures/graphs for the performance of the screws model)
├───src
├── .gitignore
├── README.md
└── main.py
```
#### Datasets
- Screws: https://universe.roboflow.com/university-of-limerick/screw_detection-ccbng/dataset/4
	- Classes:  `['Automotive Battery', 'Bike Battery', 'Dry Cell', 'Laptop Battery', 'Smartphone Battery'] `
- Batteries: https://universe.roboflow.com/project-tics-ylrlr/battery-detection-sszwf/dataset/7
	- Classes:  `['Hole', 'Screw']`

## Final Takes

### Pose Estimation
For this, a contour detection or a segmentation with appropriate edges should be performed. At the least, sort of 3D bounding box estimation should be done. There are examples of this done and to be efficient, a decent dataset including 3D objects and their real life counterparts as images should be used. Than another NN can be detected on top of the Object Detector (preferably but not a must), to catch the shape and pose of the object. A pose would be a 3D Rectangle with a possible arrow if the front of the object can be identified

### Assuming: Sky Is The Limit
I would say that, a Time of Flight camera or a Radar (short range) or a Video feed from a Camera can be utilized for an application required for **Battery Disassembly Automation**. These sensors (possibly in combination with one could give a better understanding of the depth and angle estimations. This in-turn will allow a precise understanding of the shape of the object (not just a 3D Rectangle but with the curves and edges). On top of this, a thermal camera and sensors that can measure the chemical situation on the surface of the battery can make the system safer and prevent hazardous situations. Finally, it may be possible to use these sensors to extract certain features that can also be useful within the Object Detection and Pose Estimation parts of the system.

## Appendix
Here are the links to the figures and results

> [Battery Detection Model evaluation figures: F1/PR/P_curve/R_curve/Confusion Matrices/Combined Detection/Precision/Loss etc.](https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/tree/master/runs/detect/batteries)
> 
> [Screw Detection Model evaluation figures: F1/PR/P_curve/R_curve/Confusion Matrices/Combined Detection/Precision/Loss etc.](https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/tree/master/runs/detect/screws)

Predictions (w/ changing conf. values of YOLO)
> [Predictions: Battery Detection](https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/tree/master/results/batteries)
> 
> [Predictions: Screw Detection](https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/tree/master/results/screws)
> 
> [Predictions: Combined Detection](https://github.com/enverbashirov/Object-Detection-Screws-and-Batteries/tree/master/results/combined)


