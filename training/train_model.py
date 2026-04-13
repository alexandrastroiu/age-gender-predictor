import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from prepare_dataset import load_dataset
from keras.callbacks import EarlyStopping

DATASET_PATH = "./../data/UTKFace"
MODEL_PATH = "./../models/age_gender_model.keras"
NUM_CLASSES = 5
EPOCHS = 40    
BATCH_SIZE = 32 

# Create the CNN model architecture
def build_model(shape):
    input_layer = keras.Input(shape=shape, name="Input image")
    x = layers.Conv2D(32, 3, activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x) # Reduce overfitting

    # The model has 2 outputs: predicted age category and predicted gender
    output_age = layers.Dense(NUM_CLASSES, activation="softmax", name="age_output")(x)
    output_gender = layers.Dense(1, activation="sigmoid", name="gender_output")(x)

    model = keras.Model(
        inputs=input_layer,
        outputs=[output_gender, output_age],
        name="age_gender_model"
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        loss={
            "age_output": "categorical_crossentropy",
            "gender_output": "binary_crossentropy"
        },
        loss_weights={
            "age_output": 2.0,
            "gender_output": 1.0
        },
        metrics={
            "age_output": ["accuracy"],
            "gender_output": ["accuracy"]
        }
    )

    return model

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
        shuffle=True,
        stratify=y_category
    )
    # Check shapes
    print(X_train.shape)
    print(X_test.shape)
    print(y_gender_train.shape)
    print(y_gender_test.shape)
    print(y_age_train.shape)
    print(y_age_test.shape)
    # Build the model
    model = build_model((224,224,3))
    model.summary()

    callbacks = EarlyStopping(monitor='val_age_output_accuracy',patience=5, restore_best_weights=True, mode="max")

    # Train model
    history = model.fit(
                        X_train,
                        {
                            "gender_output": y_gender_train,
                            "age_output": y_age_train
                        },
                        validation_data=(X_test,                             
                        {
                            "gender_output": y_gender_test,
                            "age_output": y_age_test
                        }),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[callbacks]
                    )
    # Evaluate model
    results = model.evaluate(X_test,
                            {
                                "gender_output": y_gender_test,
                                "age_output": y_age_test
                            })
    print(results)
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()