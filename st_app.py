import streamlit as st
import pickle as pk 
import numpy as np
import pandas as pd  # Make sure to import pandas

from st_on_hover_tabs import on_hover_tabs

st.set_page_config(page_title="MediPredict", page_icon='🏥', layout="centered", initial_sidebar_state='expanded')

# Hide the "Made with Streamlit" footer
hide_streamlit_style = """
    <style>
    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Disease Detect')

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Breast Cancer','Diabetes','Heart Disease','Kidney Disease','Liver Disease'], 
                         iconName=['monitor_heart','bar_chart_4_bars','health_and_safety','favorite','group'], default_choice=0,
                         styles={'navtab': {'background-color':'#272731',
                                            'color': '#818181',
                                            'font-size': '18px',
                                            'transition': '.3s',
                                            'white-space': 'nowrap',
                                            'text-transform': 'uppercase'},
                                 'tabOptionsStyle': {':hover :hover': {'color': 'orangered',
                                                                     'cursor': 'pointer'}},
                             },
    )


if tabs =='Diabetes':
    # Load the model and feature names only once during the app initialization
    classifier = pk.load(open('diabetes-prediction-rfc-model.pkl', 'rb'))
    feature_names = classifier['feature_names']
    # Text to display
    st.subheader("Diabetes")
    with st.form("my_form"):
        # Help text
        pregT = "Women who have had gestational diabetes during pregnancy are at \n an increased risk of developing type 2 diabetes later in life. \n Additionally, women who have had multiple pregnancies may \n also have an increased risk of developing diabetes."
        gluT = "Elevated blood glucose levels can be a sign of diabetes, and managing \n blood glucose levels is a critical component of diabetes management.\n High blood glucose levels can lead to a variety of health problems, \nincluding nerve damage, eye damage, kidney damage, \n and cardiovascular disease."
        bpT = "High blood pressure is a common comorbidity in individuals with diabetes, \n and managing blood pressure is an important aspect of diabetes management.\n High blood pressure can increase the risk of cardiovascular disease,\n kidney damage, \n and other health problems."
        skinT = "The thickness of the skin can impact the accuracy of glucose monitoring devices,\n as some devices may not be able to penetrate the \n skin of individuals with thicker skin."
        insT = "Insulin is a hormone that is critical for the regulation of blood glucose levels. \n In individuals with diabetes, the body either does not produce enough insulin or does not use insulin effectively.\n Managing insulin levels is an important aspect of diabetes management."
        bmiT = "Body mass index (BMI) is a measure of body fat based on height and weight. \n Individuals with higher BMIs are at an increased risk of developing diabetes,\n as excess body fat can make it more difficult for the body to use insulin effectively."
        dpfT = "The diabetes pedigree function is a measure of the genetic risk of developing diabetes.\n Individuals with a family history of diabetes may be at an increased risk of developing the disease."
        ageT = "Age is a risk factor for diabetes, as the risk of developing diabetes increases with age. \n Additionally, older individuals may have more difficulty managing their diabetes \n due to other health problems or complications."
        # User input:
        preg = int(st.number_input("Number of Pregnancies", min_value=0, max_value=137, step=1, help=pregT))
        glucose = int(st.number_input("Glucose Level (mg/dL)", step=1, help=gluT))
        bp = int(st.number_input("Blood Pressure (mmHg)", step=1, help=bpT))
        skinThickness = int(st.number_input("Skin Thickness mm", step=1, help=skinT))
        insulin = int(st.number_input("Insulin level (IU/mL)", step=1, help=insT))
        bmi = st.number_input("Body Mass Index (kg/m²)", help=bmiT)
        dpf = st.number_input('Diabetes Pedigree Function', help=dpfT)
        age = int(st.number_input('Patient Age in Years', step=1, help=ageT))
        submitted = st.form_submit_button("Submit")
        data = np.array([[preg, glucose, bp, skinThickness, insulin, bmi, dpf, age]])

        if submitted:
            # Assuming 'data' is the input data for prediction
            # Make sure 'data' has the same features in the same order as during training
            data_df = pd.DataFrame(data, columns=feature_names)
            my_prediction = classifier['model'].predict(data_df)
            st.title(f":blue[Prediction] :- :red[{my_prediction[0]}]", anchor=False)

if tabs =='Breast Cancer':
    st.subheader("Breast Cancer")

if tabs =='Kidney Disease':
    st.warning("Kidney Disease")

if tabs =='Heart Disease':
    st.warning("Heart Disease")

if tabs =='Liver Disease':
    st.warning("Liver Disease")
