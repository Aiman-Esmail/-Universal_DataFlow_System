import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class UniversalDataFlow:
    """
    A comprehensive data preprocessing pipeline that handles missing values,
    categorical encoding, and feature scaling.
    """
    
    def __init__(self):
        """Initialize the preprocessing components."""
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_cols = None
        self.categorical_cols = None
    
    def _identify_column_types(self, df):
        """
        Identify numeric and categorical columns in the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        """
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _impute_values(self, df):
        """
        Fill missing values using appropriate imputation strategies.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with missing values
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Impute numeric columns with median
        if self.numeric_cols:
            df_imputed[self.numeric_cols] = self.numeric_imputer.fit_transform(
                df_imputed[self.numeric_cols]
            )
        
        # Impute categorical columns with most frequent value
        if self.categorical_cols:
            df_imputed[self.categorical_cols] = self.categorical_imputer.fit_transform(
                df_imputed[self.categorical_cols]
            )
        
        return df_imputed
    
    def _encode_categorical(self, df):
        """
        Encode categorical columns using LabelEncoder.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with categorical columns to encode
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded categorical columns
        """
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def _scale_numeric(self, df):
        """
        Scale numeric columns using StandardScaler.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with numeric columns to scale
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with scaled numeric columns
        """
        df_scaled = df.copy()
        
        if self.numeric_cols:
            df_scaled[self.numeric_cols] = self.scaler.fit_transform(df_scaled[self.numeric_cols])
        
        return df_scaled
    
    def process_all(self, df):
        """
        Execute the complete data preprocessing pipeline.
        
        Performs the following steps in order:
        1. Identifies numeric and categorical columns
        2. Imputes missing values (median for numeric, most_frequent for categorical)
        3. Encodes categorical columns using LabelEncoder
        4. Scales numeric columns using StandardScaler
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame to process
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned and preprocessed DataFrame
        """
        # Identify column types
        self._identify_column_types(df)
        
        # Step 1: Impute missing values
        df_processed = self._impute_values(df)
        
        # Step 2: Encode categorical columns
        df_processed = self._encode_categorical(df_processed)
        
        # Step 3: Scale numeric columns
        df_processed = self._scale_numeric(df_processed)
        
        return df_processed
