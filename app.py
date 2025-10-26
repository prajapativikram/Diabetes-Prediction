import numpy as np
import gradio as gr
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define prediction function
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction",
    description="Predict diabetes using ML and key health parameters."
)

iface.launch()
