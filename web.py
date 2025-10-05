import os
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# -------------------- CONFIG -------------------- #
st.set_page_config(
    page_title='Disease Prediction',
    layout='wide',
    page_icon="🧑‍⚕️"
)

# -------------------- LOAD MODELS -------------------- #
BASE_DIR = os.path.dirname(__file__)
model_dir = os.path.join(BASE_DIR, "training_models")

diabetes_model = pickle.load(open(os.path.join(model_dir, "diabetes_model.sav"), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(model_dir, "heart_model.sav"), 'rb'))
parkinson_model = pickle.load(open(os.path.join(model_dir, "parkinson.sav"), 'rb'))

# -------------------- SIDEBAR MENU -------------------- #
with st.sidebar:
    selected = option_menu(
        'Prediction of Disease Outbreak System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity','heart','person'],
        default_index=0
    )

# -------------------- DIABETES -------------------- #
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        diab_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        input_df = pd.DataFrame([user_input], columns=diab_columns)
        diab_prediction = diabetes_model.predict(input_df)
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0]==1 else 'The person is not diabetic'
    st.success(diab_diagnosis)

# -------------------- HEART DISEASE -------------------- #
elif selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (0: Female, 1: Male)')
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure (mm Hg)')
    with col2:
        chol = st.text_input('Cholesterol level')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
    with col1:
        restecg = st.text_input('ECG Results')
    with col2:
        thalach = st.text_input('Max Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise-Induced Angina (1: Yes, 0: No)')
    with col1:
        oldpeak = st.text_input('ST Depression')
    with col2:
        slope = st.text_input('ST Slope')
    with col3:
        ca = st.text_input('Major Vessels')
    with col1:
        thal = st.text_input('Thalassemia')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        heart_columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        input_df = pd.DataFrame([user_input], columns=heart_columns)
        heart_prediction = heart_disease_model.predict(input_df)
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0]==1 else 'The person has no heart disease'
    st.success(heart_diagnosis)

# -------------------- PARKINSON -------------------- #
elif selected == "Parkinson Disease Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)

    inputs = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
              'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
              'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE',
              'DFA','spread1','spread2','D2','PPE']
    
    user_input = []
    cols = [col1, col2, col3, col4, col5]
    for i, feature in enumerate(inputs):
        with cols[i % 5]:
            value = st.text_input(feature)
            user_input.append(value)

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        user_input = [float(x) for x in user_input]
        input_df = pd.DataFrame([user_input], columns=inputs)
        parkinsons_prediction = parkinson_model.predict(input_df)
        parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0]==1 else "The person does not have Parkinson's disease"
    st.success(parkinsons_diagnosis)
