import streamlit as st
import pandas as pd
import joblib
import pandas as pd
import numpy as np


#Load the model
model = joblib.load('best_model.pkl')

# 1) Define mapping dicts
workclass_map = {
    'Federal-gov': 0, 'Local-gov': 1, 'Others': 2,
    'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5,
    'State-gov': 6
}

education_map = {
    '10th': 0,      '11th': 1,    '12th': 2,
    '7th-8th': 3,   '9th': 4,     'Assoc-acdm': 5,
    'Assoc-voc': 6, 'Bachelors': 7,'Doctorate': 8,
    'HS-grad': 9,   'Masters': 10,'Prof-school': 11,
    'Some-college': 12
}

marital_map = {
    'Divorced': 0, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3,
    'Married-AF-spouse': 1, 'Never-married': 4, 'Separated': 5, 'Widowed': 6
}

occupation_map = {
    'Machine-op-inspct': 6,  'Farming-fishing': 4, 'Protective-serv': 11,
    'Other-service': 7,      'Prof-specialty': 10,  'Craft-repair': 2,
    'Adm-clerical': 0,       'Exec-managerial': 3,  'Tech-support': 13,
    'Sales': 12,             'Priv-house-serv': 14, 'Transport-moving': 9,
    'Handlers-cleaners': 5,  'Armed-Forces': 1
}

relationship_map = {
    'Husband': 0, 'Not-in-family': 1, 'Own-child': 3,
    'Other-relative': 2,'Unmarried': 4,'Wife': 5
}

#race_map = {
#    'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1,
#    'Black': 2, 'Other': 3, 'White': 4
# }

gender_map = {
    'Female': 0,
    'Male': 1
}

native_country_map = {
    'United-States': 39, 'Others': 27, 'Peru': 29, 'Mexico': 25, 'Dominican-Republic': 5,
    'Ireland': 20, 'Germany': 10, 'Philippines': 30, 'Thailand': 37, 'Haiti': 13,
    'El-Salvador': 7, 'Puerto-Rico': 33, 'Vietnam': 40, 'South': 35, 'Columbia': 3,
    'Japan': 23, 'India': 18, 'Cambodia': 0, 'Poland': 31, 'England': 8,
    'Cuba': 4, 'Taiwan': 36, 'Italy': 21, 'Canada': 1, 'China': 2,
    'Nicaragua': 26, 'Honduras': 15, 'Iran': 19, 'Guatemala': 12, 'Scotland': 34,
    'Jamaica': 22, 'Portugal': 32, 'Ecuador': 6, 'Yugoslavia': 41, 'Hungary': 17,
    'Hong': 16, 'Greece': 11, 'Trinadad&Tobago': 38, 'Outlying-US(Guam-USVI-etc)': 28,
    'France': 9, 'Laos': 24, 'Holand-Netherlands': 14
}




# 2) Sidebar inputs (raw strings) create app
st.set_page_config(page_title='Employee Salary Classification', page_icon=':guardsman:', layout='centered')
st.title('Employee Salary Classification App')
st.markdown("Predict wether an employee earns >50k or <=50k based on Input features.")

#Sidebar inputs (they must match with the training data)
st.sidebar.header('Input Employee details')

# Replace with your own input fields
age = st.sidebar.slider('Age', 18, 75, 30)
workclass = st.sidebar.selectbox('Workclass', list(workclass_map.keys()))
education = st.sidebar.selectbox('Education', list(education_map.keys()))
marital_status = st.sidebar.selectbox('Marital Status', list(marital_map.keys()))
occupation = st.sidebar.selectbox('Occupation', list(occupation_map.keys()))
relationship = st.sidebar.selectbox('Relationship', list(relationship_map.keys()))
# race = st.sidebar.selectbox('Race', list(race_map.keys()))
gender = st.sidebar.selectbox('Gender', list(gender_map.keys()))
native_country = st.sidebar.selectbox('Native Country', list(native_country_map.keys()))
hours_per_week = st.sidebar.slider('Hours per week', 1, 99, 40)
experience = st.sidebar.slider('Experience', 0, 50, 10)

# ---- 3. Encode inputs ----
wc_code = workclass_map[workclass]
edu_code = education_map[education]
mar_code = marital_map[marital_status]
occ_code = occupation_map[occupation]
rel_code = relationship_map[relationship]
# race_code = race_map[race]
gen_code = gender_map[gender]
native_country_code = native_country_map[native_country]

# Filling missing columns with average values
# education_num_avg = 10.262
capital_gain_avg = 1101.502
capital_loss_avg = 88.594

# age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week 
# workclass, education, marital-status, occupation, relationship, race, gender, native-country
# Build input DataFrame (must match with the training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [wc_code],
    'education': [edu_code],
    'marital-status': [mar_code],
    'occupation': [occ_code],
    'relationship': [rel_code], 
    'gender': [gen_code],
    'native-country': [native_country_code],
    'hours-per-week': [hours_per_week],
    'experience': [experience],
    'capital-gain': [capital_gain_avg],
    'capital-loss': [capital_loss_avg],
    #'fnlwgt': [fnlwgt_avg]
    })

#Display the input DataFrame
#st.pyplot(input_df)
st.write('## Input Data:')
st.write(input_df)

# Prediction Button
if st.button('Predicting Salary Class'):
    prediction = model.predict(input_df)
    st.success(f'Predicted Salary Class: {prediction[0]}')
    st.balloons()
    st.snow()

# Batch prediction
st.markdown('---')
st.markdown('## Batch Prediction')
uploaded_file = st.file_uploader('Upload a CSV file for batch prediction', type='csv')

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Map the categorical columns to codes
    # batch_data['education'] = batch_data['education'].map(education_map)
    # batch_data['occupation'] = batch_data['occupation'].map(occupation_map)

    # Map categorical columns to codes
    # Map categorical columns
    for col, mapping in {
        'workclass': workclass_map,
        'education': education_map,
        'marital-status': marital_map,
        'occupation': occupation_map,
        'relationship': relationship_map,
        'gender': gender_map,
        'native-country': native_country_map
    }.items():
        if col in batch_data.columns:
            batch_data[col] = batch_data[col].map(mapping)

    # Fill missing columns with average values
    #batch_data['education-num'] = education_num_avg
    batch_data['capital-gain'] = capital_gain_avg
    batch_data['capital-loss'] = capital_loss_avg
    #batch_data['fnlwgt'] = fnlwgt_avg

    # Replace NaNs caused by unmapped categories
    batch_data = batch_data.fillna(-1)

    st.write('Uploaded Data Preview:')
    st.write('Uploded Data Preview:', batch_data.head())

    # Make predictions
    batch_preds = model.predict(batch_data)
    batch_data['Predicted Class'] = batch_preds

    # st.write('Batch Prediction Results:')
    # st.write(batch_data)
    st.write("Batch Predictions:")
    st.write(batch_data.head())

    # Download predictions
    st.write("Download Predictions:")
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button(label='Download Predictions', data=csv, file_name='predictions.csv', mime='text/csv')
