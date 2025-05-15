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

input_data = {
    'CreditScore': [credit_score],
    'Gender' : [le.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary],
    'Geography' : [geography]
}




## preprocess the data


def preprocess_input_dictionary(input_data):
    df = pd.DataFrame(input_data)
    # df['Gender'] = le.transform(df['Gender'])
    df.reset_index(drop=True, inplace=True)
    encoded_geo = ohe.transform(df[['Geography']]).toarray()
    encoded_geo_df = pd.DataFrame(encoded_geo, columns=ohe.get_feature_names_out(['Geography']))
    encoded_geo_df.reset_index(drop=True, inplace=True)
    df.drop(['Geography'], axis=1, inplace=True) 
    df = pd.concat([df, encoded_geo_df], axis=1)
    df = scaler.transform(df)
    return df

input_data_scaled = preprocess_input_dictionary(input_data)

## predict the churn 

if st.button(label='Predict Customer Churn'):

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Prediction Probability: {prediction_proba:.2f}')
    # Display the prediction

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn')
