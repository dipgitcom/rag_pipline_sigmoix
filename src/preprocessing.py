# src/preprocessing.py
import fitz  # PyMuPDF
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep financial notation
        text = re.sub(r'[^\w\s\$\%\.\,\(\)\-\+]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into chunks with metadata"""
        clean_text = self.clean_text(text)
        chunks = self.text_splitter.split_text(clean_text)
        
        return [
            {
                "content": chunk,
                "chunk_id": i,
                "length": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]