from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank_results(self, query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank retrieval results using cross-encoder"""
        if not results:
            return results
        
        # Prepare query-document pairs
        pairs = [(query, result["content"]) for result in results]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add reranking scores to results
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by reranking score
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]