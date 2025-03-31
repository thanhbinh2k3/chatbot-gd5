from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import openai

# Load API Key từ .env
load_dotenv()
api_key = os.getenv("KEY_API_GPT")  # Đảm bảo tên biến đúng với .env

if not api_key:
    raise ValueError("❌ LỖI: API Key không tìm thấy! Kiểm tra lại .env")

# ✅ Đúng cú pháp OpenAI API >= 1.0.0
client = openai.OpenAI(api_key=api_key)

app = FastAPI()

# Cấu hình CORS cho phép truy cập từ trình duyệt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model nhận dữ liệu từ người dùng
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "✅ API Chatbot Trường Phái Hội Họa GPT-4o-mini đang chạy!"}

@app.post("/chatbot")
def chatbot_response(request: ChatRequest):
    user_message = request.message.strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="❌ Tin nhắn không được để trống!")

    try:
        # Gọi OpenAI API với cú pháp mới
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là một AI chuyên về trường phái hội họa!"},
                {"role": "user", "content": user_message}
            ]
        )

        bot_reply = response.choices[0].message.content.strip()

    except Exception as e:
        bot_reply = f"🚨 Đã xảy ra lỗi: {str(e)}"

    return {"response": bot_reply}
#Chạy server bằng lệnh:
#uvicorn chatbot.main:app --reload --host 127.0.0.1 --port 8000 
# # Mo web
# http://localhost:8000/