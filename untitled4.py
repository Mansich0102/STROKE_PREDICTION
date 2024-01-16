# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 00:46:57 2024

@author: mansi
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 00:14:12 2024

@author: mansi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:30:27 2024

@author: mansi
"""

import sklearn 
import numpy as np 
import pickle
import streamlit as st

load_model=pickle.load(open("C:/Users/mansi/OneDrive/Documents/CAPSTONE_PROJECT_DATASETS/stroke2_model.sav",'rb'))

#creating a function

def stroke_prediction(input_data):
    # Convert input data to numeric values
    input_data = [float(value) if isinstance(value, str) else value for value in input_data]
    np_array=np.asarray(input_data)
    ip_reshaped=np_array.reshape(1,-1)
    prediction=load_model.predict(ip_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return'The individual does not have a stroke'
    else: 
        return"The individual has had a stroke"

def main():
    
    # give a title
    st.title('Stroke Prediction Web App')
    
    # getting input data from user
    
    
    gender=st.text_input("Your Gender type")
    age=st.text_input("Your age")
    hypertension=st.text_input("Do you have hypertension or not")
    heart_disease=st.text_input("Do you have any type of heart disease or not")
    ever_married=st.text_input("Your Married status")
    work_type=st.text_input("Your Work Type/ Profession")
    Residence_type=st.text_input("Your Residence Type")
    avg_glucose_level=st.text_input("Your Average glucose level")
    bmi=st.text_input(" Your BMI")
    smoking_status=st.text_input("Your Smoking Status")
    
    #code for prediction
    diagnosis='' 
    
    #button
    
    if st.button('stroke test result'):
        diagnosis=stroke_prediction([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status])
    st.success(diagnosis)  
    

if __name__=='__main__':
    main()

    

