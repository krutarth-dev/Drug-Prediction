import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from pickle import dump
from pickle import load
import pickle


# Reading the data 
data=pd.read_csv("train_data.csv",header=None)
df = data.iloc[:,1:]
Y=data.iloc[:,0:1]

# Changing the column names of the train dataset to start from 0
for i in df.columns:
    df.rename(columns={i:(i-1)},inplace=True)
    
# Removing all the columns for which the Nan values are greater than 15% of total train data
for col in df.columns:
    if df[col].isna().sum()> 0.15*(df.shape[0]):
        df.drop(columns=col,inplace=True)

        
# Replacing the null values with the Mean of that feature
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(df)
df = imputer.transform(df)
df=pd.DataFrame(df)


# Scaling the Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df)
df = scaler.transform(df)


# Dimension Reduction using pca
from sklearn.decomposition import PCA
pca = PCA()
train_pca_values = pca.fit_transform(df)
X = pd.DataFrame(train_pca_values[:,:100])


# SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=9)
x_train_sm,y_train_sm = sm.fit_resample(X,Y)


# Loading the model
loaded_model = load(open('dt 2.sav', 'rb'))


# creating a function for Prediction
def activity_prediction(input_data):
    
    # Reading training data 
    train_X = pd.read_csv("train_data.csv",header=None)
    train_X = train_X.iloc[:,1:]
    train_Y = train_X.iloc[:,0]
    
    # Changing the column names of the train dataset to start from 0
    for i in train_X.columns:
        train_X.rename(columns={i:(i-1)},inplace=True)
    
    # Temporarily Combining train and input data for ease of data preprocessing
    combined_data = pd.concat([train_X,input_data])
    
    # Removing all the columns for which the Nan values are greater than 15% of total train data
    for col in combined_data.columns:
        if combined_data[col].isna().sum()> 0.15*(combined_data.shape[0]):
            combined_data.drop(columns=col,inplace=True)
    
    # Replacing the null values with the Mean of that feature
    B = combined_data
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer = imputer.fit(B)
    B = imputer.transform(B)
    df_nonna = pd.DataFrame(B)
    
    # Scaling the data 
    input_data_scaled=scaler.transform(df_nonna)
    
    # Separating train data and input data
    train_X = pd.DataFrame(input_data_scaled[:800,:])
    in_data = pd.DataFrame(input_data_scaled[800:,:])
    
    # Applying PCA
    test_pca_values = pca.transform(in_data)
    final_data= pd.DataFrame(test_pca_values[:,:100])
    
    # Making predictions
    prediction = loaded_model.predict(final_data)
    print(prediction)

    if (prediction[0] == 1):
      return 'Active'
    else:
      return 'Inactive'
  
    
  
def main():
    
    # giving a title
    st.title('Drug Activity Prediction')
    
    # getting the input data from the user
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file,header=None)
        st.write('Input data')
        st.write(dataframe)
        
    
    # code for Prediction
    result = ''
    
    # creating a button for Prediction
    if st.button('Predict'):
        result = activity_prediction(dataframe)
    if result=="Active":   
        st.success(result)
    else:
        st.error(result)
    
    
if __name__ == '__main__':
    main()