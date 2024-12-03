# src/data_exploration.py

import pandas as pd

def heart_attack_risk_distribution(df):
    """Calculate the distribution of Heart Attack Risk."""
    return round(100 * df['Heart_Attack_Risk'].value_counts() / df.shape[0], 2)

def age_distribution(df):
    """Describe the Age distribution."""
    return df['Age'].describe()

def sex_distribution(df):
    """Calculate the distribution of Sex."""
    return round(100 * df['Sex'].value_counts() / df.shape[0], 2)

def correlation_matrix(df):
    """Compute the Pearson correlation matrix for numeric columns."""
    
    # Select only numeric columns from the DataFrame
    numeric_df = df.select_dtypes(include=[float, int])
    
    # Compute and return the Pearson correlation matrix
    return numeric_df.corr(method='pearson')