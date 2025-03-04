import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit App Title
st.title("Iris Flower Species Predictor")
st.write("Enter the petal & sepal measurements below to predict the Iris species.")

# User Inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

# Convert inputs into a DataFrame
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=['sepal length (cm)', 'sepal width (cm)', 
                                   'petal length (cm)', 'petal width (cm)'])

# Standardize input using the saved scaler
input_scaled = scaler.transform(input_data)

# Predict species
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    species_dict = {0: "Setosa", 1: "Versicolor ", 2: "Virginica"}
    st.success(f"Predicted Species: {species_dict[prediction]}")
