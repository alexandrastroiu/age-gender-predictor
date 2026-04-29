# Age & Gender Prediction App

## Overview

Desktop application that performs real-time face detection and analysis in order to predict age and gender category using a Convolutional Neural Network (CNN). 

## Features
* Real-time face detection using Haar Cascade Classifiers
* Gender and age category prediction using a CNN model trained on the UTKFace dataset
* Interactive Tkinter-based graphical user interface
* Share prediction results via mail app

## Technologies used
* Python
* OpenCV  - image processing and face detection
* Keras   - deep learning model
* Tkinter - graphical user interface

## Running the application

1. Clone the repository
```
git clone https://github.com/alexandrastroiu/age-gender-predictor.git
cd age-gender-predictor/
```

2. Create virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Save the trained model 
```
python train_model.py 
```

4. Run the application
```
python tkinter_ui.py
```