import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def configure_model():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    return model
