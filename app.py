import streamlit as st
import joblib
import numpy as np 

st.title("Salary Prediction App")

st.divider()

st.write("With this app, you can get estimations for the salaries of ")

years = st.number_input("Enter the years at company",value = 1, step=1, min_value=0)
jobrate = st.number_input("Enter the job rate", value= 3.5, step = 0.5, min_value=0.0)

X = [years,jobrate]

model = joblib.load("linearmodel.pkl")

st.divider()

predict = st.button("Press the button for salary prediciton")

st.divider()

if predict:

    st.balloons()

    X1 = np.array([X])

    prediction = model.predict(X1)[0]

    st.write(f"Salary prediciton is {prediction:,.2f}")

else:
    "Please press the button for app to make the prediciton"
