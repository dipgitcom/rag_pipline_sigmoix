# src/query_optimizer.py
from typing import List, Dict
import openai

class QueryOptimizer:
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
    
    def optimize_query(self, original_query: str) -> Dict[str, str]:
        """Optimize query for better retrieval"""
        prompt = f"""Given the following financial query, rewrite it to be more specific and likely to retrieve relevant information from financial documents:

Original query: {original_query}

Please provide:
1. An optimized version that includes relevant financial terms
2. Alternative phrasings that might capture the same information
3. Key financial concepts to look for

Format your response as JSON with keys: "optimized", "alternatives", "key_concepts"
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            import json
            return json.loads(response.choices[0].message.content)
        except:
            return {
                "optimized": original_query,
                "alternatives": [original_query],
                "key_concepts": []
            }