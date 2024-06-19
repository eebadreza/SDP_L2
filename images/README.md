# Dataset Folders : Validation and Training

This repository contains two folders for managing datasets related to diabetic retinopathy detection.

## Validation Folder

The `validation` folder contains images categorized into 5 classes for validating diabetic retinopathy detection models.

### Classes

The images are organized into the following classes:

- **Mild:** Images showing mild diabetic retinopathy.
- **Moderate:** Images showing moderate diabetic retinopathy.
- **Severe:** Images showing severe diabetic retinopathy.
- **Proliferate:** Images showing proliferative diabetic retinopathy.
- **NO_DR:** Images showing no signs of diabetic retinopathy.

Each class folder contains images representative of that specific category, aiding in the validation and evaluation of diabetic retinopathy detection models.

### Dataset Information

#### Dataset Link

The images in this folder are sourced from the following dataset:

- [Diabetic Retinopathy 224x224 Gaussian Filtered Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered/data?select=train.csv)

#### About the Dataset

This dataset provides preprocessed retinal images categorized into various stages of diabetic retinopathy. It includes images resized to 224x224 pixels and filtered with Gaussian blur for enhanced feature detection.

The dataset is useful for training and validating machine learning models aimed at detecting diabetic retinopathy based on retinal images.

---

## Training Folder

The `training` folder contains a large collection of training images for diabetic retinopathy detection.

### Folder Contents

The training images are categorized into different classes similar to the validation set, but due to size limitations, these images could not be included in this repository.

### Note

To access the complete training dataset, download it from the following source:

- [Diabetic Retinopathy 224x224 Gaussian Filtered Dataset on Kaggle](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)

After downloading, place the images into the `training` folder within this repository. This will allow you to use a comprehensive set of images for robust model development and evaluation.

---

Feel free to customize this README file with additional details or adjust the formatting as needed for your project.
