import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        """Perform automated data cleaning."""
        # 1. Remove duplicate rows
        self.df.drop_duplicates(inplace=True)
        
        # 2. Handle missing values
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Fill missing text with 'Unknown'
                self.df[col] = self.df[col].fillna("Unknown")
            else:
                # Fill missing numbers with the average (Mean)
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        return self.df

    def get_info(self):
        """Return basic statistics about the processed data."""
        return {
            "rows": len(self.df),
            "cols": len(self.df.columns),
            "nulls": self.df.isnull().sum().sum()
        }
