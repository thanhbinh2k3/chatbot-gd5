import openai
import os
from dotenv import load_dotenv

# Load API Key từ .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Chatbot:
    def __init__(self):
        pass  # Không cần tải mô hình về máy

    def get_response(self, query: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Bạn là một bác sĩ AI chuyên về y học."},
                    {"role": "user", "content": query}
                ]
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Lỗi: {str(e)}"
