# Additional requirements
# tabula-py==2.8.2
# pdfplumber==0.9.0
# sqlalchemy==2.0.23

# src/structured_processor.py
import pandas as pd
import pdfplumber
import tabula
from typing import List, Dict, Any
import json

class StructuredDataProcessor:
    def __init__(self):
        self.tables = []
        self.structured_data = {}
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables using multiple methods"""
        tables = []
        
        # Method 1: pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:  # Has headers
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df['source_page'] = page_num + 1
                        tables.append(df)
        
        # Method 2: tabula (fallback)
        try:
            tabula_tables = tabula.read_pdf(pdf_path, pages='all')
            tables.extend(tabula_tables)
        except Exception as e:
            print(f"Tabula extraction failed: {e}")
        
        return tables
    
    def process_financial_tables(self, tables: List[pd.DataFrame]) -> Dict[str, Any]:
        """Process and categorize financial tables"""
        processed_data = {
            "income_statement": [],
            "balance_sheet": [],
            "cash_flow": [],
            "other_metrics": []
        }
        
        for i, table in enumerate(tables):
            # Clean table
            table = self.clean_table(table)
            
            # Categorize based on content
            table_text = ' '.join(table.astype(str).values.flatten()).lower()
            
            if any(term in table_text for term in ['revenue', 'income', 'expense']):
                processed_data["income_statement"].append({
                    "table_id": i,
                    "data": table.to_dict('records'),
                    "columns": list(table.columns)
                })
            elif any(term in table_text for term in ['assets', 'liabilities', 'equity']):
                processed_data["balance_sheet"].append({
                    "table_id": i,
                    "data": table.to_dict('records'),
                    "columns": list(table.columns)
                })
            elif any(term in table_text for term in ['cash flow', 'operating', 'investing']):
                processed_data["cash_flow"].append({
                    "table_id": i,
                    "data": table.to_dict('records'),
                    "columns": list(table.columns)
                })
            else:
                processed_data["other_metrics"].append({
                    "table_id": i,
                    "data": table.to_dict('records'),
                    "columns": list(table.columns)
                })
        
        return processed_data
    
    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize table data"""
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df

class HybridRetriever(VectorRetriever):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.structured_data = {}
    
    def add_structured_data(self, structured_data: Dict[str, Any]):
        """Add structured data for hybrid search"""
        self.structured_data = structured_data
    
    def search_structured_data(self, query: str) -> List[Dict]:
        """Search structured data using keyword matching"""
        results = []
        query_lower = query.lower()
        
        for category, tables in self.structured_data.items():
            for table_info in tables:
                # Search in column names and data
                columns_text = ' '.join(table_info['columns']).lower()
                data_text = json.dumps(table_info['data']).lower()
                
                # Simple keyword matching (can be enhanced with fuzzy matching)
                if any(word in columns_text or word in data_text for word in query_lower.split()):
                    results.append({
                        "category": category,
                        "table_id": table_info['table_id'],
                        "data": table_info['data'],
                        "columns": table_info['columns'],
                        "relevance_score": self._calculate_keyword_score(query_lower, columns_text + data_text)
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:3]  # Top 3
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """Simple keyword relevance scoring"""
        query_words = query.split()
        matches = sum(1 for word in query_words if word in text)
        return matches / len(query_words)
    
    def hybrid_retrieve(self, query: str, top_k: int = 3) -> Dict[str, List]:
        """Perform hybrid retrieval combining vector and structured search"""
        # Vector search for text
        text_results = self.retrieve(query, top_k)
        
        # Structured search for tables
        structured_results = self.search_structured_data(query)
        
        return {
            "text_context": text_results,
            "structured_data": structured_results
        }