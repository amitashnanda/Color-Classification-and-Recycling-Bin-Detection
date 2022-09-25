# Color-Classification-and-Recycling-Bin-Detection

## Introduction
<p align = "center">
This repository shows the developmenet of a color classification model and drawing the concept from later to detect recycle bins using Gaussian Discriminant Analysis. Data is essential while dealing with any Machine Learning approach and concepts. One of the crucial aspects of
data is color. It is an important characteristic of an object because of its significant influence on value. Hence, getting the color parameter into consideration is an utmost priority while dealing with classification models.
</p>

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
<p align = "center">
We implemented our model on classifying the blue pictures on the validation dataset with an accuracy of 100 % (precision = 1). While on the unknown images we obtained a score of 9.934/10. On successful implementation of our model on bin detection for the ten validation datasets, we get an accuracy of 90%. While tested on the unknown images gives a score of 7.18/10. The inaccuracy on the unknown images can be attributed to the cases where some images may contain blue colour bin like object but not exactly the blue bin region. So, this may be too big as we did not set any bounding box on the maximum area of the bounding box.
</p>
  

<!-- ![Alt text](https://github.com/[amitashnanda]/[Color-Classification-and-Recycling-Bin-Detection]/blob/[results]/Figure_1a.png?raw=true)
![Alt text](https://github.com/[amitashnanda]/[Color-Classification-and-Recycling-Bin-Detection]/blob/[results]/Figure_1.png?raw=true) -->
![Boundary Box](/results/Figure_1a.png)
![Bin detection](/results/Figure_1.png)
![Boundary Box](/results/Figure_2a.png)
![Bin detection](/results/Figure_2a.png)
![Boundary Box](/results/Figure_3a.png)
![Bin detection](/results/Figure_3a.png)
![Boundary Box](/results/Figure_4a.png)
![Bin detection](/results/Figure_4.png)





