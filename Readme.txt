# README: Loading and Running the Skin Lesion Segmentation Code

## Project Overview
This README provides instructions on how to load and run the Attention U-Net code for skin lesion segmentation using the ISIC 2018 dataset.

## Requirements
To run the code, you'll need the following:
- **Python 3.x**
- **TensorFlow 2.x**
- **Albumentations**
- **Matplotlib**
- **Pillow**
- **Google Colab** (recommended for easy access to Google Drive)

## Steps to Load and Run the Code

### 1. **Set Up Google Colab**
- Open Google Colab in your browser.
- Mount Google Drive to access the dataset by running the following code snippet:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Make sure the ISIC 2018 dataset is available in your Google Drive.

### 2. **Install Requirements**
- Install the necessary libraries by running the following commands in a code cell:
  ```python
  !pip install albumentations
  !pip install Pillow
  ```
- Import the required libraries:
  ```python
  import numpy as np
  import tensorflow as tf
  from tensorflow.keras import layers
  import os
  import matplotlib.pyplot as plt
  from PIL import Image
  from tensorflow.keras.applications import EfficientNetB0
  from tensorflow.keras.layers import *
  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  import albumentations as A
  from functools import partial
  from tqdm import tqdm
  ```

### 3. **Load the Dataset**
- Define the paths for the dataset:
  ```python
  train_img_folder_path = "/content/drive/MyDrive/archive (2)/ISIC2018_Task1-2_Training_Input"
  train_label_folder_path = "/content/drive/MyDrive/archive (2)/ISIC2018_Task1_Training_GroundTruth"
  val_folder_path = "/content/drive/MyDrive/archive (2)/ISIC2018_Task1-2_Validation_Input"
  test_folder_path = "/content/drive/MyDrive/archive (2)/ISIC2018_Task1-2_Test_Input"
  ```
- Load the dataset paths for training, validation, and testing:
  ```python
  train_images_path = np.sort([os.path.join(train_img_folder_path, i) for i in os.listdir(train_img_folder_path) if i.endswith(('.jpg','.png'))])
  train_labels_path = np.sort([os.path.join(train_label_folder_path, i) for i in os.listdir(train_label_folder_path) if i.endswith(('.jpg','.png'))])
  test_images_path = np.sort([os.path.join(test_folder_path, i) for i in os.listdir(test_folder_path) if i.endswith(('.jpg','.png'))])
  ```

### 4. **Prepare the Data**
- Split the training data into training and validation sets (80-20 split):
  ```python
  train_split = 0.8
  val_images_path = train_images_path[int(train_split*len(train_images_path)):]
  val_labels_path = train_labels_path[int(train_split*len(train_labels_path)):]
  train_images_path = train_images_path[:int(train_split*len(train_images_path))]
  train_labels_path = train_labels_path[:int(train_split*len(train_labels_path))]
  ```

### 5. **Create and Compile the Model**
- Build the Attention U-Net model by running the provided model architecture code.
- Compile the model using the following command:
  ```python
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=bce_dice_loss_log,
                metrics=[dice_coef, iou, 'accuracy', specificity, sensitivity])
  ```

### 6. **Train the Model**
- Train the model using the training dataset:
  ```python
  history = model.fit(train_ds, epochs=35, verbose=2,
                      steps_per_epoch=steps_per_epoch,
                      validation_steps=val_steps,
                      validation_data=val_ds,
                      callbacks=callbacks)
  ```

### 7. **Evaluate and Predict**
- Make predictions on the test dataset:
  ```python
  raw_predictions = model.predict(test_ds)
  ```
- Save the predicted segmentation masks as PNG images:
  ```python
  if not os.path.exists('predictions'):
      os.makedirs('predictions')

  for i, pred in tqdm(enumerate(predictions)):
      pred = (pred * 255).astype(np.uint8)
      image = Image.fromarray(pred.squeeze())
      filename = f"prediction_{i}.png"
      image.save(os.path.join('predictions', filename))
  ```

## How to Run
1. **Open Google Colab** and set up your environment by mounting Google Drive.
2. **Install necessary packages** using the commands provided.
3. **Load and split the dataset** as per the instructions.
4. **Build, compile, and train the model**.
5. **Evaluate** the model and **save predictions**.

Make sure to follow each step carefully to successfully run the code and obtain segmentation results.