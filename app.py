import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained Random Forest model
with open('RandomForestModel.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load the cleaned dataset
df = pd.read_csv('Cleaned_data.csv')

# Define the Streamlit app
def main():
    st.title("Diabetes Prediction App")

    # Create input fields for user
    pregnancies = st.number_input("Enter the number of Pregnancies:", min_value=0.0,max_value=17.0)
    glucose = st.number_input("Enter the Glucose level:", min_value=0.0, max_value=200.0)
    blood_pressure = st.number_input("Enter the Blood Pressure:", min_value=0.0, max_value=150.0)
    skin_thickness = st.number_input("Enter the Skin Thickness:", min_value=0.0, max_value=100.0)
    insulin = st.number_input("Enter the Insulin Level:", min_value=0.0, max_value=1000.0)
    bmi = st.number_input("Enter the BMI:", min_value=0.0, max_value= 100.0)
    diabetes_pedigree_function = st.number_input("Enter the Diabetes Pedigree Function:", min_value=0.0, max_value=5.0)
    age = st.number_input("Enter the Age:", min_value=0.0)

    # Create a feature DataFrame from user inputs
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Predict using the model
    if st.button("Predict"):
        prediction = rf_model.predict(input_data)
        if prediction[0] == 0:
            st.write("The person is not diabetic")
        else:
            st.write("The person is diabetic")

if __name__ == '__main__':
    main()
