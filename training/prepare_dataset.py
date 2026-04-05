import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
SAMPLES =  5000 # modify sample size for training

def load_dataset(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Path to dataset not found: {dataset_path}")
    
    files = os.listdir(dataset_path)
    random.shuffle(files)
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
            print(f"An error ocurred while accessing file: {file}")

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


def plot_age_distribution(ages):
    sns.histplot(ages, bins=20, kde=True, color='lightgreen', edgecolor='blue')
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.ylabel('Distribution')
    plt.show()

def plot_category_distribution(categories):
    counter = Counter(categories)
    labels = counter.keys()
    values = counter.values()
    plt.bar(labels, values)
    plt.title("Category distribution")
    plt.show()

def plot_gender_distribution(genders):
    counter = Counter(genders)
    labels = ["Male", "Female"]
    values = [counter[0], counter[1]]
    plt.pie(values, labels=labels)
    plt.title("Gender distribution")
    plt.show()

# Test
def main():
    X, y_gender, y_age, y_category = load_dataset(DATASET_PATH)

    print("Images shape:", X.shape)
    print("Gender labels shape:", y_gender.shape)
    print("Age labels shape:", y_age.shape)
    print("Category labels shape:", y_category.shape)

    plot_age_distribution(y_age)
    plot_category_distribution(y_category)
    plot_gender_distribution(y_gender)
    
if __name__ == "__main__":
    main()