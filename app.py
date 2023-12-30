import streamlit as st
import pickle as pk 
import joblib
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
        #----xx-------
        submitted = st.form_submit_button("Predict Diabetes")
        data = np.array([[preg, glucose, bp, skinThickness, insulin, bmi, dpf, age]])

        if submitted:
            # Make sure 'data' has the same features in the same order as during training
            data_df = pd.DataFrame(data, columns=feature_names)
            my_prediction = classifier['model'].predict(data_df)
            st.title(f":blue[Prediction] :- :red[{my_prediction[0]}]", anchor=False)

if tabs =='Breast Cancer':
    # Load the model and feature names only once during the app initialization
    brest_cancer_model = pk.load(open('model.pkl', 'rb'))
    st.subheader("Breast Cancer")
    
    with st.form("Brest Cancer"):
        # Helper Text needs to be added
        cct = "This refers to the thickness of the cell clusters in the sample. \nThe value ranges from 1 to 10, with 1 being a thin cluster and 10 being a very thick cluster."
        ucs = "This measures the consistency of cell size in the sample.\n A value of 1 indicates that the cells are all roughly the same size, while a value of 10 indicates a high degree of variation in cell size."
        ucsh = "This measures the consistency of cell shape in the sample.  \nA value of 1 indicates that the cells are all roughly the same shape, while a value of 10 indicates a high degree of variation in cell shape."
        ma = "This refers to the degree of adhesion (i.e. sticking together) of the cells at the edges of the cell clusters. \n A value of 1 indicates that the cells have very little adhesion, while a value of 10 indicates that they are highly adhesive."
        ses = "This measures the size of the individual cells in the sample. \n A value of 1 indicates that the cells are small, while a value of 10 indicates that they are large."
        bn = "This refers to the presence or absence of a nucleus in the cells. \n A value of 1 indicates that there are no nuclei, while a value of 10 indicates that there are many nuclei."
        bb = "This measures the uniformity of staining of the chromatin (the material that makes up the cell nucleus).  \nA value of 1 indicates that the staining is uniform, while a value of 10 indicates that there is a high degree of variation in staining."
        nn = "This measures the size and shape of the nucleoli (small structures inside the nucleus that produce ribosomes). \n A value of 1 indicates that the nucleoli are small and uniform, while a value of 10 indicates that they are large and irregular."
        mi = "This measures the number of cells in the sample that are undergoing mitosis (cell division). \n A value of 1 indicates that there are few cells undergoing mitosis, while a value of 10 indicates that there are many cells undergoing mitosis."
        #user input:
        clump_thickness = int(st.number_input("Clump Thickness",step=1,help=cct))
        uniform_cell_size = int(st.number_input("Uniform Cell Size",step=1,help=ucs))
        uniform_cell_shape = int(st.number_input("uniform cell shape",step=1,help=ucsh))
        marginal_adhesion = int(st.number_input("Marginal Adhesion",step=1,help=ma))
        single_epithelial_size = int(st.number_input("Single Epithelial Size",step=1,help=ses))
        bare_nuclei = int(st.number_input("Bare Nuclei",step=1,help=bn))
        bland_chromatin = int(st.number_input("Bland Chromatin",step=1,help=bb))
        normal_nucleoli = int(st.number_input("Normal Nucleoli",step=1,help=nn))
        mitoses = int(st.number_input("Mitoses",step=1,help=mi))
        submitted = st.form_submit_button("Predict Cancer")
        # Input List
        input_features = [clump_thickness, uniform_cell_size, uniform_cell_shape, marginal_adhesion,
                single_epithelial_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]
        features_value = [np.array(input_features)]
        # Features Names
        features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                        'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
        bcd = pd.DataFrame(features_value, columns=features_name)
        # st.dataframe(bcd)
        if submitted:
            output = brest_cancer_model.predict(bcd)
            if output == 4:
                st.error("a high risk of Breast Cancer")
            else:
                st.error("a low risk of Breast Cancer")
#---------not working---------
def Kidney_Disease_Predictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

if tabs =='Kidney Disease':
    st.subheader("Kidney Disease")
    with st.form("Brest Cancer"):
        #user input:
        blood_pressure =  int(st.number_input("Blood Pressure:",step=1))
        specific_gravity = int(st.number_input("Specific Gravity",step=1))
        albumin =int(st.number_input("Albumin",step=1))
        blood_sugar_level =int(st.number_input("Blood Sugar Level",step=1))
        red_blood_cells_count = int(st.number_input("Red Blood Cells Count",step=1))
        pus_cell_count =int(st.number_input("Pus Cell Count",step=1))
        pus_cell_clumps = int(st.number_input("Pus Cell Clumps",step=1))
        submitted = int(st.form_submit_button("Predict Kidney Disease"))
        # Create a dictionary
        user_inputs = {
            "blood_pressure": blood_pressure,
            "specific_gravity": specific_gravity,
            "albumin": albumin,
            "blood_sugar_level": blood_sugar_level,
            "red_blood_cells_count": red_blood_cells_count,
            "pus_cell_count": pus_cell_count,
            "pus_cell_clumps": pus_cell_clumps
        }
        to_predict_list = list(user_inputs.values())
        to_predict_list = list(map(float, to_predict_list))
        if submitted:
            if len(to_predict_list) == 7:
                result = Kidney_Disease_Predictor(to_predict_list, 7)

            if(int(result) == 1):
                prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
                st.error(prediction)
            else:
                prediction = "Patient has a low risk of Kidney Disease"
                st.success(prediction)
if tabs =='Heart Disease':
    st.warning("Heart Disease")

if tabs =='Liver Disease':
    st.warning("Liver Disease")
