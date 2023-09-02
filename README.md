# Histopathological Image Analysis for Cancer Diagnosis

![Project Image](dataset/demo/1.jpeg)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains the code and resources for an automated system developed for diagnosing lung and colon cancer from histopathological images. The project utilizes machine learning techniques to analyze medical images and provide accurate diagnostic results.

## Project Overview

- Developed an automated system for cancer diagnosis using machine learning.
- Implemented a multi-class Convolutional Neural Network (CNN) architecture.
- Achieved an impressive overall accuracy of 91.05% in cancer diagnosis.
- Emphasized ethical considerations, including data anonymization and algorithm usage with medical consultation.

## Key Features

- Accurate classification of lung and colon cancer types.
- High overall accuracy in distinguishing between benign and malignant cases.
- Ethical considerations for patient data privacy.
- Potential for further model enhancements and medical applications.

## Technologies Used

- Python
- Jupyter Notebook
- Kaggle dataset [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images)
- Machine Learning libraries (e.g., TensorFlow, scikit-learn)

## Installation

1. Clone this repository.
   ``` git clone https://github.com/swapguru/Lung-Colon-Cancer-Detection.git ```
2. Install the required dependencies.
   ``` pip install -r requirements.txt ```

## Usage
  - Run the Jupyter Notebook to train and test the machine learning model.
  - Provide input histopathological images for diagnosis.
  - Edit start.py to include histopathological image path:
    ```
    img0 = 'dataset\\demo\\1.jpeg'
    ```
  - Run start.py:
    ```
    python start.py
    ```
  - Console output:
    ```
    Select one of the following options:
    1. Demo Image 1 # Labeled as Lung - Malignant - SCC
    2. Demo Image 2 # Labeled as Colon - Benign
    3. Demo Image 3 # Labeled as Colon - Malignant
    4. Demo Image 4 # Labeled as Lung - Benign
    5. Demo Image 5 # Labeled as Lung - Malignant - ACA
    1
    Predicted as Lung: Malignant - SCC
    ```

## Results
  - Overall accuracy of 91.05% in cancer diagnosis.
  - Detailed analysis of classification results.
  - Ethical considerations and guidelines for using the model.

## Project Structure
 - `data/`: Contains the dataset used for training and testing.
 - `notebooks/`: Jupyter Notebooks for model development and evaluation.
 - `src/`: Source code for the machine learning model.
 - `docs/`: Additional documentation or resources (if any).
   
## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:
 - Fork the repository.
 - Create a new branch for your feature/bugfix.
 - Make your changes and submit a pull request.

## License
  This project is licensed under the MIT License.

Disclaimer: This project is for educational and research purposes only. It should not replace professional medical diagnosis and consultation.
