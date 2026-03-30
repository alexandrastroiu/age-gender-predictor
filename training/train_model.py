import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from prepare_dataset import load_dataset

DATASET_PATH = "./../data/UTKFace"
NUM_CLASSES = 7

# Create the CNN model build architecture
def build_model():
    pass

def main():
    # Load dataset
    X, y_gender, y_age, y_category = load_dataset(DATASET_PATH)
    y_category_encoded = to_categorical(y_category, num_classes = NUM_CLASSES)
    # Split data into train and test data
    X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
        X,
        y_gender,
        y_category_encoded,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    # Check shapes
    print(X_train.shape)
    print(X_test.shape)
    print(y_gender_train.shape)
    print(y_gender_test.shape)
    print(y_age_train.shape)
    print(y_age_test.shape)
    # Build the model
    model = model_build()
    # Train model
    # Evaluate model
    # Save model


if __name__ == "__main__":
    main()