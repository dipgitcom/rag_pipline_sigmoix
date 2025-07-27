# src/generation.py
import openai
from typing import List, Dict

class FinancialQAGenerator:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        if api_key:
            openai.api_key = api_key
        self.model = model
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        # Combine context
        context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        
        prompt = f"""Based on the following context from financial reports:

{context}

Answer the query: {query}

Please provide a direct, factual answer based only on the information provided in the context. If the information is not available in the context, state that clearly."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, factual answers based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"