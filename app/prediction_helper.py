from joblib import load
import pandas as pd
# from pyarrow.compute import scalar


model_rest=load("artifacts\model__rest.joblib")
model_young=load("artifacts\model_young.joblib")
scalar_rest=load("artifacts\scaler_rest.joblib")
scalar_young=load("artifacts\scaler_young.joblib")


def calculate_normalized_risk(medical_history):
    # Individual disease risk scores (from original training encoding)
    disease_risk = {
        'no disease': 0,
        'thyroid': 5,
        'diabetes': 6,
        'high blood pressure': 6,
        'heart disease': 8,
        'none': 0
    }

    # Split combined medical history into individual diseases
    medical_history_split = {
        'No Disease': ('no disease', 'none'),
        'Diabetes': ('diabetes', 'none'),
        'High blood pressure': ('high blood pressure', 'none'),
        'Thyroid': ('thyroid', 'none'),
        'Heart disease': ('heart disease', 'none'),
        'Diabetes & High blood pressure': ('diabetes', 'high blood pressure'),
        'Diabetes & Thyroid': ('diabetes', 'thyroid'),
        'Diabetes & Heart disease': ('diabetes', 'heart disease'),
        'High blood pressure & Heart disease': ('high blood pressure', 'heart disease'),
    }

    # Get disease pair
    disease_1, disease_2 = medical_history_split.get(
        medical_history, ('none', 'none'))

    # Calculate total risk score
    total_risk_score = disease_risk[disease_1] + disease_risk[disease_2]

    # Apply same min-max normalization as training
    min_score = 0  # min from training data
    max_score = 14  # max from training data

    normalized = (total_risk_score - min_score) / (max_score - min_score)

    return normalized



def preprocess_input(input_dict):
    expected_cols=['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
       'genetical_risk', 'normalized_risk_score', 'gender_Male',
       'region_Northwest', 'region_Southeast', 'region_Southwest',
       'marital_status_Unmarried', 'bmi_category_Obesity',
       'bmi_category_Overweight', 'bmi_category_Underweight',
       'smoking_status_Occasional', 'smoking_status_Regular',
       'employment_status_Salaried', 'employment_status_Self-Employed']


    # insurance plan encoded
    insurance_plan_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

   #
    df = pd.DataFrame(0,columns=expected_cols,index=[0])

    # ✅ Set numeric columns directly
    df['age'] = input_dict['age']
    df['number_of_dependants'] = input_dict['number_of_dependants']
    df['income_lakhs'] = input_dict['income_lakhs']
    df['genetical_risk'] = input_dict['genetical_risk']

    # ✅ Ordinal encoding for insurance plan
    df['insurance_plan'] = insurance_plan_map[input_dict['insurance_plan']]



    #----------

    # ----------------------
    # ✅ Normalized risk score
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['medical_history'])
    df=handle_scaling(input_dict['age'],df)



    # ✅ One-hot encoding
    df['gender_Male'] = 1 if input_dict['gender'] == 'Male' else 0
    df['region_Northwest'] = 1 if input_dict['region'] == 'Northwest' else 0
    df['region_Southeast'] = 1 if input_dict['region'] == 'Southeast' else 0
    df['region_Southwest'] = 1 if input_dict['region'] == 'Southwest' else 0
    df['marital_status_Unmarried'] = 1 if input_dict['marital_status'] == 'Unmarried' else 0
    df['bmi_category_Obesity'] = 1 if input_dict['bmi_category'] == 'Obesity' else 0
    df['bmi_category_Overweight'] = 1 if input_dict['bmi_category'] == 'Overweight' else 0
    df['bmi_category_Underweight'] = 1 if input_dict['bmi_category'] == 'Underweight' else 0
    df['smoking_status_Occasional'] = 1 if input_dict['smoking_status'] == 'Occasional' else 0
    df['smoking_status_Regular'] = 1 if input_dict['smoking_status'] == 'Regular' else 0
    df['employment_status_Salaried'] = 1 if input_dict['employment_status'] == 'Salaried' else 0
    df['employment_status_Self-Employed'] = 1 if input_dict['employment_status'] == 'Self-Employed' else 0

    return df
    # Drop original categorical columns
    df.drop(columns=['gender', 'region', 'marital_status', 'bmi_category',
                     'smoking_status', 'employment_status', 'medical_history'], inplace=True)

    return df[expected_cols]

def handle_scaling(age,df):
    if age<=25:
        scaler_object = scalar_young
    else:
        scaler_object = scalar_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df[cols_to_scale]=scaler.transform(df[cols_to_scale])

    return df




def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['age']<=25:
        prediction=model_young.predict(input_df)
    else:
        prediction=model_rest.predict(input_df)

    return int(prediction)
