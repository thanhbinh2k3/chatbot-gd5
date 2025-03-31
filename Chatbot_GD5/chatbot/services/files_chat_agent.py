import faiss
import numpy as np
import logging
from chatbot.utils.llm import LLM  # noqa: I001
from chatbot.utils.retriever import Retriever
from chatbot.utils.document_grader import DocumentGrader
from chatbot.utils.answer_generator import AnswerGenerator
from chatbot.utils.no_answer_handler import NoAnswerHandler

from langgraph.graph import END, StateGraph, START
from chatbot.utils.graph_state import GraphState
from typing import Dict, Any
from app.config import settings
import os

class FilesChatAgent:
    """
    Lớp FilesChatAgent chịu trách nhiệm quản lý quy trình chatbot,
    từ tìm kiếm tài liệu, đánh giá độ liên quan đến tạo câu trả lời và xuất kết quả HTML.
    """

    def __init__(self, path_vector_store: str) -> None:
        """
        Khởi tạo FilesChatAgent với các thành phần chính.

        Args:
            path_vector_store (str): Đường dẫn đến thư mục lưu trữ vector store.
        """
        # Kiểm tra và đảm bảo chỉ mục FAISS tồn tại
        faiss_index_path = os.path.join(path_vector_store, "index.faiss")
        
        if not os.path.exists(faiss_index_path):
            logging.error(f"Tệp chỉ mục FAISS không tồn tại tại: {faiss_index_path}")
            raise FileNotFoundError(f"Tệp chỉ mục FAISS không tồn tại tại: {faiss_index_path}")

        try:
            self.retriever = Retriever(settings.LLM_NAME).set_retriever(path_vector_store)  # Khởi tạo trình tìm kiếm tài liệu
        except Exception as e:
            logging.error(f"Lỗi khi khởi tạo retriever: {e}")
            raise RuntimeError(f"Lỗi khi khởi tạo retriever: {e}")
        
        self.llm = LLM().get_llm(settings.LLM_NAME)  # Khởi tạo mô hình ngôn ngữ
        self.document_grader = DocumentGrader(self.llm)  # Bộ đánh giá tài liệu
        self.answer_generator = AnswerGenerator(self.llm)  # Bộ tạo câu trả lời
        self.no_answer_handler = NoAnswerHandler(self.llm)  # Xử lý trường hợp không có câu trả lời

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        Tìm kiếm các tài liệu liên quan đến câu hỏi.

        Args:
            state (GraphState): Trạng thái hiện tại chứa câu hỏi.

        Returns:
            dict: Chứa danh sách tài liệu và câu hỏi.
        """
        question = state["question"]
        documents = self.retriever.get_documents(question, int(settings.NUM_DOC))
        return {"documents": documents, "question": question}

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        Tạo câu trả lời dựa trên các tài liệu liên quan.

        Args:
            state (GraphState): Trạng thái hiện tại chứa câu hỏi và tài liệu.

        Returns:
            dict: Chứa câu trả lời đã được tạo.
        """
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(doc.page_content for doc in documents)  # Ghép nội dung các tài liệu thành một đoạn văn
        generation = self.answer_generator.get_chain().invoke({"question": question, "context": context})
        return {"generation": generation}

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Xác định xem có nên tạo câu trả lời hay không dựa trên tài liệu tìm được.

        Args:
            state (GraphState): Trạng thái hiện tại chứa danh sách tài liệu.

        Returns:
            str: "no_document" nếu không có tài liệu, "generate" nếu có thể tạo câu trả lời.
        """
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---QUYẾT ĐỊNH: KHÔNG CÓ VĂN BẢN LIÊN QUAN ĐẾN CÂU HỎI, BIẾN ĐỔI TRUY VẤN---")
            return "no_document"
        else:
            print("---QUYẾT ĐỊNH: TẠO CÂU TRẢ LỜI---")
            return "generate"

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Chấm điểm tài liệu để xác định mức độ liên quan.

        Args:
            state (GraphState): Trạng thái hiện tại chứa câu hỏi và danh sách tài liệu.

        Returns:
            dict: Chứa danh sách tài liệu đã lọc và câu hỏi.
        """
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            score = self.document_grader.get_chain().invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print("---CHẤM ĐIỂM: TÀI LIỆU LIÊN QUAN---")
                filtered_docs.append(d)
            else:
                print("---CHẤM ĐIỂM: TÀI LIỆU KHÔNG LIÊN QUAN---")

        return {"documents": filtered_docs, "question": question}

    def handle_no_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        Xử lý trường hợp không tìm thấy câu trả lời phù hợp.

        Args:
            state (GraphState): Trạng thái hiện tại chứa câu hỏi.

        Returns:
            dict: Chứa câu trả lời mặc định hoặc phản hồi phù hợp.
        """
        question = state["question"]
        generation = self.no_answer_handler.get_chain().invoke({"question": question})
        return {"generation": generation}

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của chatbot, bao gồm các bước tìm kiếm, đánh giá và tạo câu trả lời.

        Returns:
            StateGraph: Đồ thị trạng thái của quy trình chatbot.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve)  # Bước tìm kiếm tài liệu
        workflow.add_node("grade_documents", self.grade_documents)  # Bước chấm điểm tài liệu
        workflow.add_node("generate", self.generate)  # Bước tạo câu trả lời
        workflow.add_node("handle_no_answer", self.handle_no_answer)  # Bước xử lý khi không có tài liệu

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "no_document": "handle_no_answer",
                "generate": "generate",
            },
        )

        workflow.add_edge("generate", END)

        return workflow

# Hàm kiểm tra và tạo chỉ mục FAISS nếu không có
def create_faiss_index(path_vector_store: str):
    """
    Kiểm tra nếu không có tệp index.faiss thì tạo mới.
    """
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

# Gọi hàm tạo chỉ mục FAISS nếu chưa có
create_faiss_index("D:/Thuc_Tap/Tuần 5/Chatbot_GD5/demo/data_vector")
