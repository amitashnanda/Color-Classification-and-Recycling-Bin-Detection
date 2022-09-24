# Color-Classification-and-Recycling-Bin-Detection

## Introduction
This repository shows the developmenet of a color classification model and drawing the concept from later to detect recycle bins using Gaussian Discriminant Analysis. Data is essential while dealing with any Machine Learning approach and concepts. One of the crucial aspects of
data is color. It is an important characteristic of an object because of its significant influence on value. Hence, getting the color parameter into consideration is an utmost priority while dealing with classification models.

## Folder Structure

```
Color-Classification-and-Recycling-Bin-Detection
│   README.md
│   requirements.text  
└───bin_detection
│   │   data
|   |   |   training
|   |   └───validation
|   |   parameters
|   |   roipoly
|   |   |   roiply.py
|   |   └───version.py
│   └───src
|       |   bin_detector.py
|       |   test_bin_detector.py
|       |   test_roipoly.py
|       └───generate_rgb_values.py
└───pixel_classification
|    |   data
|    |   |   training
|    |   └───validation
|    |   parameters
|    └───src
|        |   generate_rgb_data.py
|        |   pixel_classifier.py
|        └───test_pixel_classifier.py
|
└───docs 
└───results
│
```

## Prerequisites
The dependencies are listed under requirements.txt and are all purely python based. To install them simply run
```
pip install -r requirements.txt
```

## Dataset
The datset is there inside each data folder.

## Running
For pixel classification 
```
python test_pixel_classifier.py
```
For bin detection
```
python test_bin_detector.py
```
## Results
![alt text](https://github.com/[amitashnanda]/[Color-Classification-and-Recycling-Bin-Detection]/blob/[results]/Figure_1a.png?raw=true)
![alt text](https://github.com/[amitashnanda]/[Color-Classification-and-Recycling-Bin-Detection]/blob/[results]/Figure_1.png?raw=true)


