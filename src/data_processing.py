# src/data_processing.py

import pandas as pd

def split_blood_pressure(df):
    """Split 'Blood Pressure' column into 'Systolic Blood Pressure' and 'Diastolic Blood Pressure'."""
    # Split the 'Blood Pressure' column and ensure the values are numeric
    bp_split = df['Blood_Pressure'].str.split('/', expand=True)
    df['Systolic_Blood_Pressure'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['Diastolic_Blood_Pressure'] = pd.to_numeric(bp_split[1], errors='coerce')
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataframe by returning a count of missing values for each column."""
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

def encode_features(df):
    """Encode categorical features into numerical values and drop unnecessary columns."""
    # Encoding categorical variables
    df['Diet'] = df['Diet'].replace({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})
    df['Sex'] = df['Sex'].replace({'Male': 0, 'Female': 1})
    df['Continent'] = df['Continent'].replace({
        'South America': 0, 'Africa': 1, 'Asia': 2, 
        'Europe': 3, 'North America': 4, 'Australia': 5
    })
    df['Hemisphere'] = df['Hemisphere'].replace({
        'Southern Hemisphere': 0, 'Northern Hemisphere': 1
    })

    # Dropping unnecessary columns
    df = df.drop(columns=['Patient_ID', 'Blood_Pressure', 'Country', 'Heart_Attack_Risk'], errors='ignore')
    return df

def preprocess_test_data(df_test, X_train_columns):
    """Preprocess the test dataset to match the structure and encoding of the training data."""
    # Split 'Blood Pressure' column
    bp_split = df_test['Blood_Pressure'].str.split('/', expand=True)
    df_test['Systolic_Blood_Pressure'] = pd.to_numeric(bp_split[0], errors='coerce')
    df_test['Diastolic_Blood_Pressure'] = pd.to_numeric(bp_split[1], errors='coerce')

    # Store 'Patient ID' column for final submission
    test_id = df_test['Patient_ID']

    # Drop unnecessary columns
    df_test = df_test.drop(columns=['Patient_ID', 'Blood_Pressure', 'Country'], errors='ignore')

    # Encode categorical features
    df_test['Diet'] = df_test['Diet'].replace({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})
    df_test['Sex'] = df_test['Sex'].replace({'Male': 0, 'Female': 1})
    df_test['Continent'] = df_test['Continent'].replace({
        'South America': 0, 'Africa': 1, 'Asia': 2, 
        'Europe': 3, 'North America': 4, 'Australia': 5
    })
    df_test['Hemisphere'] = df_test['Hemisphere'].replace({
        'Southern Hemisphere': 0, 'Northern Hemisphere': 1
    })

    # Ensure column order matches the training data
    df_test = df_test.reindex(columns=X_train_columns, fill_value=0)
    
    return df_test, test_id
