import pandas as pd
import numpy as np

def process_data(df):
    report = []
    
    # 1. Missing Values Handling
    initial_nulls = df.isnull().sum().sum()
    if initial_nulls > 0:
        # Fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        report.append(f"Fixed {initial_nulls} missing values using median/mode imputation.")
    else:
        report.append("No missing values detected.")

    # 2. Duplicate Removal
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        report.append(f"Removed {duplicates} duplicate rows to ensure data integrity.")

    # 3. Imbalance Detection (Standard AI Requirement)
    # Assuming the last column is the target/label
    target_col = df.columns[-1]
    if df[target_col].nunique() < 10:  # Check if it's a classification task
        counts = df[target_col].value_counts()
        imbalance_ratio = counts.max() / counts.min()
        
        if imbalance_ratio > 1.5:
            report.append(f"ALERT: Significant Class Imbalance detected in '{target_col}' (Ratio: {imbalance_ratio:.2f}:1).")
            report.append("System Recommendation: Apply SMOTE or Random Oversampling before training.")
        else:
            report.append(f"Class distribution in '{target_col}' is balanced.")

    return df, report
