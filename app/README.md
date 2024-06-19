# Diabetic Retinopathy Detection App

This folder contains the code and resources for the Diabetic Retinopathy Detection App. The app is designed to detect diabetic retinopathy using image classification models.

## Folder Structure

- `bgs`: Contains backgrounds used in the app.
- `model`: Contains the models used in the app. Specifically, it uses MobileNet models saved in the `pretrained_models` folder in the parent directory.
- `main.py`: The main page code for the app.
- `requirements.txt`: Lists the dependencies required to run the app.
- `util.py`: Contains background processes and classification utilities.

### bgs

The `bgs` folder contains background images that are used in the app for visual presentation and user interface design.

### model

The `model` folder contains the models used for classifying images. These models are based on MobileNet and are saved in the `pretrained_models` folder in the previous directory.

### main.py

The `main.py` file is the main entry point for the app. It includes the code for the app's user interface, handling user inputs, and displaying results.

### requirements.txt

The `requirements.txt` file lists all the dependencies needed to run the app. Install these dependencies using pip:

# util.py

The `util.py` file contains background processes and functions for image classification. This includes loading models, preprocessing images, and making predictions.

## Result

The app demonstrates high accuracy in binary classification (i.e., DR and NO_DR). However, it shows room for improvement in 5-class classification, which could benefit from a more sophisticated model.

## Reference

This project was inspired by the following reference: [YouTube Video](https://www.youtube.com/watch?v=n_eMARPqBZI)
