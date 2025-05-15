import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle 
import tensorflow as tf 

# Load the trained models
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scalers

with open('label_encoder_gender.pkl', 'rb') as f:
    le = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


## streamlit app

st.title("Customer Churn Prediction")

# user input

geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = ({
    'CreditScore': [credit_score],
    'Gender' : [le.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# encode geography 
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = ohe.get_feature_names_out(['Geography']))

input_data = pd.DataFrame(input_data)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# now apply scaling
input_data_scaled = scaler.transform(input_data)

# Predict churn

if st.button(label='Predict Customer Churn'):

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.' + ' with probability of : ' + str(prediction_proba))
    else:
        st.write('The customer is not likely to churn' + ' with probability of : ' + str(prediction_proba))
