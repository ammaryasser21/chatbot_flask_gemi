from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import os
import re
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load API key
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from environment variables")

genai.configure(api_key=GENAI_API_KEY)

# Define response model
class MedicalResponse(BaseModel):
    is_medical: bool
    message: str

# دالة لتحديد لغة ولهجة المستخدم باستخدام Gemini AI
def detect_language_and_dialect(text: str) -> str:
    prompt = f"""
    Identify the language and dialect of the following text. 
    Provide the response in the format: 
    {{"language": "Language Name", "dialect": "Dialect Name"}}
    
    Text: "{text}"
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    try:
        response_text = re.sub(r"```json\n?|```", "", response.text).strip()
        lang_data = json.loads(response_text)
        return lang_data["language"], lang_data.get("dialect", "Standard")
    except json.JSONDecodeError:
        return "Unknown", "Unknown"

def is_medical_question(prompt: str) -> bool:
    check_prompt = f"""
    You are an AI medical assistant. Determine if the following user input is related to a medical condition, symptoms, treatment, or diagnosis.
    Answer ONLY "yes" or "no".
    User Input: "{prompt}"
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(check_prompt)
    return response.text.strip().lower() == "yes"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip()

    if not prompt:
        return jsonify({"error": "Message is required"}), 400

    medical = is_medical_question(prompt)

    if not medical:
        response = MedicalResponse(is_medical=False, message="This query is not related to a medical issue.")
        return jsonify(response.dict())

    # تحديد اللغة واللهجة تلقائيًا
    language, dialect = detect_language_and_dialect(prompt)

    try:
        structured_prompt = f"""
        You are an AI medical assistant. Generate a short, user-friendly, and comprehensive response explaining the patient's issue, including possible causes and suggested next steps.
        The response should be in the "{language}" language and follow the "{dialect}" dialect, ensuring it matches the user's way of speaking.
        
        Response format:
        {{"message": "Detailed but concise explanation, covering causes and next steps."}}

        User Input: "{prompt}"
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(structured_prompt)
        response_text = re.sub(r"```json\n?|```", "", response.text).strip()
        response_data = json.loads(response_text)

        validated_response = MedicalResponse(is_medical=True, message=response_data["message"])
        return jsonify(validated_response.dict())

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from AI"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
