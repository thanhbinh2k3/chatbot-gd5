import os
import fitz  # PyMuPDF để đọc PDF
import pandas as pd  # Pandas để đọc CSV
from docx import Document  # python-docx để đọc DOCX
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from ingestion.service_manager import ServiceManager
from langchain_community.vectorstores import FAISS

class Ingestion:
    """
    Lớp thực hiện quá trình ingest dữ liệu từ các tệp văn bản vào vector store.
    """

    def __init__(self, embedding_model_name: str):
        self.chunk_size = 2000  # Kích thước đoạn văn bản tối đa
        self.chunk_overlap = int(self.chunk_size * 0.2)  # Độ chồng chéo
        self.embedding_model = ServiceManager().get_embedding_model(embedding_model_name)

    def ingestion_folder(self, path_input_folder: str, path_vector_store: str):
        all_docs = []
        
        for root, _, files in os.walk(path_input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith(".txt"):
                    docs = self.process_txt(file_path)
                elif file.endswith(".pdf"):
                    docs = self.process_pdf(file_path)
                elif file.endswith(".csv"):
                    docs = self.process_csv(file_path)
                elif file.endswith(".docx"):
                    docs = self.process_docx(file_path)
                else:
                    continue  # Bỏ qua file không hỗ trợ
                
                all_docs.extend(docs)

        # Ensure the vector store directory exists
        if not os.path.exists(path_vector_store):
            os.makedirs(path_vector_store)  # Tạo thư mục nếu chưa tồn tại

        # Tạo vector store
        vectorstore = FAISS.from_documents(all_docs, self.embedding_model)

        # Save FAISS index to file (index.faiss will be created here)
        vectorstore.save_local(path_vector_store)

    def process_txt(self, path_file: str):
        documents = TextLoader(path_file, encoding="utf8").load()
        text = "\n".join([doc.page_content for doc in documents])  # Chuyển về string
        return self.split_documents(text, path_file)

    def process_pdf(self, path_file: str):
        with fitz.open(path_file) as doc:
            text = "\n".join([page.get_text("text") for page in doc])
        return self.split_documents(text, path_file)

    def process_csv(self, path_file: str):
        df = pd.read_csv(path_file, encoding="utf8")
        text = "\n".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))
        return self.split_documents(text, path_file)

    def process_docx(self, path_file: str):
        doc = Document(path_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return self.split_documents(text, path_file)

    def split_documents(self, text, path_file):
        if not isinstance(text, str):  # Đảm bảo text luôn là string
            text = "\n".join(text) if isinstance(text, list) else str(text)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", ","],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = text_splitter.create_documents([text])
        
        for doc in docs:
            doc.metadata["file_name"] = os.path.basename(path_file)
            doc.metadata["chunk_size"] = self.chunk_size
        
        return docs
