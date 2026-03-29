import cv2
import os
import random
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
    files = random.shuffle(files)
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
            images.append(preprocessed_image)
            genders.append(gender)
            ages.append(age)
            categories.append(age_class)
        except Exception as e:
            print("An error ocurred while accessing file: f{file}")

        X = np.array(images)
        y_gender = np.array(genders)
        y_age = np.array(ages)
        y_category = np.array(categories)

        return X, y_gender, y_age, y_category


    

def preprocess_image(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    image = cv2.resize(image, SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    return image

def get_labels(filename):
    root, ext  = os.path.splitext(filename)
    labels = root.split("_")

    if len(labels) != 4:
        raise ValueError(f"Invalid file format: {filename}")
    
    age = int(labels[0])
    gender = int(labels[1])
    category = get_category(age)
    return age, gender, category

def get_category(age):
    for index, category in enumerate(AGE_CATEGORY):
        if category[0] <= age and age <= category[1]:
            return index
    
    raise ValueError(f"Age {age} does not belong in any category")



def get_age_distribution():
    pass

def get_category_distribution():
    pass

def get_gender_distribution():
    pass