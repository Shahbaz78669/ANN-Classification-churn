import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pickle

model=tf.keras.models.load_model('model.h5')

with open('lable_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


with open('onehot_encoder_geography.pkl','rb') as file:
    onehot_encoder_geography=pickle.load(file)


with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 40)
balance = st.number_input('Balance', min_value=0.0, value=60000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure (years)', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


sample_input = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }



sample_input_df = pd.DataFrame([sample_input])

    # Encode Gender
sample_input_df['Gender'] = label_encoder_gender.transform(sample_input_df['Gender'])



geo_encoded = onehot_encoder_geography.transform(sample_input_df[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

    # Combine
sample_input_df = pd.concat([sample_input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

    # Scale
input_scaled = scaler.transform(sample_input_df)

    # Predict
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write(f"**Churn Probability:** {prediction_prob:.2%}")

if prediction_prob > 0.5:
        st.error("The customer is likely to churn ")
else:
        st.success("The customer is likely to stay ")