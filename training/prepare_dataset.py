import cv2
import os
import numpy as np

DATASET_PATH = "./../data/UTKFace"
SIZE = (224, 224)
AGE_CATEGORY = [
    (0,12),
    (13, 19),
    (20, 29),
    (30, 45),
    (46, 60),
    (60, 75),
    (75, 120)
]
SAMPLES = 5000

def load_dataset(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Path to dataset not found: {dataset_path}")
    
    files = os.listdir(dataset_path)
    files = files[:SAMPLES]
    
    images = []
    genders = []
    ages = []
    categories = []

    for file in files:
        fpath = f"{dataset_path}/{file}"

        try:
            age, gender, age_class = get_labels(file)
            image = cv2.imread(fpath)
            if image is None:
                raise ValueError("Cannot read image")
            preprocessed_image = preprocess_image(image)
            images.append(preprocess_image)
            genders.append()
            ages.append()
            categories.append()
        except Exception as e:
            print("An error ocurred while accessing file: f{file}")

        X = np.array(images)
        y_gender = np.array(genders)
        y_age = np.array(ages)
        y_category = np.array(categories)

        return X, y_gender, y_age, y_category


    

def preprocess_image():
    pass

def get_labels():
    pass

def get_age_distribution():
    pass

def get_category_distribution():
    pass

def get_gender_distribution():
    pass