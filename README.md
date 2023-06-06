# Drug-Prediction


## Summary:
The project focuses on drug activity prediction using machine learning techniques. It involves building a classification model to classify compounds as active or inactive. The code provided utilizes Python and various libraries such as Pandas, NumPy, Scikit-learn, and Streamlit for data preprocessing, model training, and deployment.

## Project Description:
The drug prediction project aims to develop a predictive model that can determine the activity of compounds. The project utilizes a dataset containing information about various compounds and their corresponding activity labels. Here's an overview of the steps involved:

### 1. Data Preprocessing:
The provided code starts by reading the dataset using Pandas and separating the features (input data) and labels (activity) into separate dataframes. It performs data cleaning by removing columns with a high percentage of missing values. Missing values in other columns are replaced with the mean value of each feature. The data is then scaled using StandardScaler and dimensionality reduction is performed using PCA (Principal Component Analysis).

### 2. Handling Class Imbalance:
Since the dataset may have imbalanced classes, the code applies the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to balance the classes by oversampling the minority class.

### 3. Model Training: 
The code loads a pre-trained Decision Tree Classifier model (commented out in the provided code). However, it seems that the loading of the model is not implemented. You may need to uncomment and properly load the trained model to make predictions.

### 4. Prediction Function:
The code defines a function called `activity_prediction` that takes input data, preprocesses it, applies the trained model, and returns the predicted activity (active or inactive).

### 5. Deployment:
The code uses the Streamlit library to create a web-based interface for the drug activity prediction. It allows users to upload a file containing input data, and upon clicking the "Predict" button, the model predicts the activity and displays the result (active or inactive) on the interface.
