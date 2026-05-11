import os
import streamlit as st
import pandas as pd
import google.generativeai as genai

# Configure Google Gemini API
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API Key not found. Please check Environment Variables in Render.")

# Initialize the model with the correct name for version 1.5
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def main():
    st.set_page_config(page_title="Universal DataFlow System", layout="wide")
    
    st.title("Universal DataFlow System")
    st.subheader("Automated Data Cleaning & AI Analysis")

    # File Upload Section
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File Loaded Successfully!")
            
            # Display Data Preview
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Data Statistics
            if st.checkbox("Show Data Summary"):
                st.write(df.describe())

            # AI Analysis Section
            st.write("### AI Data Assistant")
            user_question = st.text_input("Ask a question about your data:")

            if user_question:
                # Prepare a brief context for the AI
                data_context = f"Dataset columns: {', '.join(df.columns.tolist())}. Data shape: {df.shape}."
                prompt = f"Context: {data_context}\nQuestion: {user_question}"
                
                try:
                    response = model.generate_content(prompt)
                    st.write("### AI Response:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error connecting to AI: {str(e)}")

        except Exception as e:
            st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    main()
