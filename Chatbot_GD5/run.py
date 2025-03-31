import os
from ingestion.ingestion import Ingestion
from chatbot.services.files_chat_agent import FilesChatAgent
from app.config import settings
import logging

# Đảm bảo rằng thư mục lưu trữ vectorstore tồn tại
path_vector_store = "D:/Thuc_Tap/Tuần 5/Chatbot_GD5/demo/data_vector"
if not os.path.exists(path_vector_store):
    os.makedirs(path_vector_store)

# Kiểm tra và tạo chỉ mục FAISS nếu chưa có
def create_faiss_index(path_vector_store: str):
    faiss_index_path = os.path.join(path_vector_store, "index.faiss")
    if not os.path.exists(faiss_index_path):
        logging.info(f"Tạo chỉ mục FAISS tại: {faiss_index_path}")
        
        # Tạo dữ liệu mẫu (embedding)
        embeddings = np.random.random((1000, 128)).astype('float32')  # Giả sử có 1000 tài liệu với vector 128 chiều
        
        # Tạo chỉ mục FAISS
        index = faiss.IndexFlatL2(128)  # 128 là chiều dài của vector
        index.add(embeddings)  # Thêm các embedding vào chỉ mục

        # Lưu chỉ mục FAISS vào tệp
        faiss.write_index(index, faiss_index_path)
        logging.info(f"Chỉ mục FAISS đã được tạo thành công tại {faiss_index_path}")
    else:
        logging.info(f"Chỉ mục FAISS đã tồn tại tại {faiss_index_path}")

# Tạo FAISS index nếu chưa có
create_faiss_index(path_vector_store)

# Sau khi đã tạo hoặc xác nhận FAISS index tồn tại, tiến hành ingest dữ liệu
Ingestion("openai").ingestion_folder(
     path_input_folder="D:/Thuc_Tap/Tuần 5/Chatbot_GD5/demo/data_in",
     path_vector_store=path_vector_store,
)

# Sử dụng FilesChatAgent với thư mục vector store đã có chỉ mục FAISS
_question = "Chủ nghĩa biểu hiện (Expressionism) là gì? Chủ nghĩa biểu hiện (Expressionism) xuất hiện vào thời gian nào và trong bối cảnh nào?"
chat = FilesChatAgent(path_vector_store).get_workflow().compile().invoke(
    input={
        "question": _question,
    }
)

print(chat)
print("generation", chat["generation"])
