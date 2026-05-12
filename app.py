import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize API key and Client
# This setup ensures your API key remains private
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("Critical Error: GROQ_API_KEY is not set in the environment.")

client = Groq(api_key=api_key)

def get_ai_insight(data_summary, sample_rows):
    """
    Sends data context to Llama-3 model for professional analysis.
    """
    prompt = f"""
    You are a professional Data Analyst.
    
    ### Dataset Preview (First 5 Rows):
    {sample_rows}
    
    ### Statistical Summary:
    {data_summary}
    
    ### Analysis Goal:
    1. Identify hidden trends within the variables.
    2. Suggest potential areas for predictive modeling.
    3. Highlight any data quality issues or outliers.
    """

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "Provide technical and concise analysis in professional English."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"AI Generation Error: {str(e)}"

def analyze_dataset(file_path):
    """
    Handles CSV reading and initiates AI processing.
    """
    try:
        # Load dataset using pandas
        df = pd.read_csv(file_path)
        
        # Prepare summaries for the LLM
        stats = df.describe(include='all').to_string()
        preview = df.head().to_string()
        
        print(f"Dataset loaded: {len(df)} rows detected.")
        
        # Request AI Analysis
        return get_ai_insight(stats, preview)

    except Exception as e:
        return f"Data Load Error: {str(e)}"

if __name__ == "__main__":
    # Specify the target data file
    # Ensure 'data.csv' is in your project directory
    target_csv = 'data.csv' 
    
    if os.path.exists(target_csv):
        report = analyze_dataset(target_csv)
        print("\n" + "="*30)
        print(" AUTOMATED DATA REPORT ")
        print("="*30 + "\n")
        print(report)
    else:
        print(f"Error: The file '{target_csv}' was not found.")