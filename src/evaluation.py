# src/evaluation.py
import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
import json

class RAGEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_retrieval(self, retrieved_chunks: List[Dict], relevant_chunks: List[int]) -> Dict[str, float]:
        """Evaluate retrieval quality"""
        retrieved_ids = [chunk.get("chunk_id", -1) for chunk in retrieved_chunks]
        
        # Precision@k
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_chunks))
        precision_at_k = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        
        # Recall@k
        recall_at_k = relevant_retrieved / len(relevant_chunks) if relevant_chunks else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_chunks:
                mrr = 1 / (i + 1)
                break
        
        return {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr
        }
    
    def evaluate_answer_quality(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Evaluate answer quality using ROUGE and BLEU"""
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
        
        # BLEU score
        reference_tokens = reference_answer.split()
        generated_tokens = generated_answer.split()
        bleu_score = sentence_bleu([reference_tokens], generated_tokens)
        
        return {
            "rouge1_f": rouge_scores['rouge1'].fmeasure,
            "rouge2_f": rouge_scores['rouge2'].fmeasure,
            "rougeL_f": rouge_scores['rougeL'].fmeasure,
            "bleu": bleu_score
        }
    
    def factual_accuracy_score(self, generated_answer: str, ground_truth_facts: List[str]) -> float:
        """Simple factual accuracy check"""
        found_facts = 0
        for fact in ground_truth_facts:
            if fact.lower() in generated_answer.lower():
                found_facts += 1
        
        return found_facts / len(ground_truth_facts) if ground_truth_facts else 0

# Advanced RAG Pipeline
class AdvancedRAGPipeline:
    def __init__(self, api_key: str = None):
        self.query_optimizer = QueryOptimizer(api_key)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator = HybridQAGenerator(api_key)
        self.evaluator = RAGEvaluator()
        
    def process_query(self, query: str, use_optimization: bool = True, use_reranking: bool = True) -> Dict:
        """Process query through advanced RAG pipeline"""
        # Step 1: Query optimization
        if use_optimization:
            optimized_queries = self.query_optimizer.optimize_query(query)
            search_query = optimized_queries["optimized"]
        else:
            search_query = query
        
        # Step 2: Retrieval
        hybrid_results = self.hybrid_retriever.hybrid_retrieve(search_query, top_k=5)
        
        # Step 3: Reranking
        if use_reranking and hybrid_results["text_context"]:
            hybrid_results["text_context"] = self.reranker.rerank_results(
                search_query, 
                hybrid_results["text_context"], 
                top_k=3
            )
        
        # Step 4: Generation
        answer = self.generator.generate_hybrid_answer(
            query,
            hybrid_results["text_context"],
            hybrid_results["structured_data"]
        )
        
        return {
            "original_query": query,
            "search_query": search_query,
            "answer": answer,
            "text_context": hybrid_results["text_context"],
            "structured_data": hybrid_results["structured_data"]
        }