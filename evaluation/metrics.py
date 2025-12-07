"""
Evaluation Metrics Module
Implements various evaluation metrics for RAG systems
"""

from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import re

import numpy as np

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Collection of evaluation metrics for RAG systems"""
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[Any],
        relevant_ids: List[Any],
        k: int = 10,
    ) -> float:
        """
        Calculate Recall@k
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Cutoff position
            
        Returns:
            Recall@k score
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
        return recall
    
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[Any],
        relevant_ids: List[Any],
        k: int = 10,
    ) -> float:
        """
        Calculate Precision@k
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision@k score
        """
        if not retrieved_ids[:k]:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        precision = len(retrieved_at_k & relevant_set) / k
        return precision
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids_list: List[List[Any]],
        relevant_ids_list: List[List[Any]],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_ids_list: List of retrieved document ID lists
            relevant_ids_list: List of relevant document ID lists
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            relevant_set = set(relevant)
            
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[Any],
        relevant_ids: List[Any],
        relevance_scores: Dict[Any, float] = None,
        k: int = 10,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@k)
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            relevance_scores: Optional dict of relevance scores (binary if None)
            k: Cutoff position
            
        Returns:
            nDCG@k score
        """
        def dcg(ids, scores, k):
            dcg_score = 0.0
            for i, doc_id in enumerate(ids[:k], 1):
                rel = scores.get(doc_id, 0.0)
                dcg_score += rel / np.log2(i + 1)
            return dcg_score
        
        # Use binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}
        
        # Calculate DCG
        dcg_score = dcg(retrieved_ids, relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_ids = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
        idcg_score = dcg(ideal_ids, relevance_scores, k)
        
        if idcg_score == 0:
            return 0.0
        
        return dcg_score / idcg_score
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """
        Calculate Exact Match score
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        def normalize_text(text):
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        
        pred_norm = normalize_text(prediction)
        gt_norm = normalize_text(ground_truth)
        
        return 1.0 if pred_norm == gt_norm else 0.0
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        def get_tokens(text):
            text = text.lower()
            tokens = re.findall(r'\w+', text)
            return Counter(tokens)
        
        pred_tokens = get_tokens(prediction)
        gt_tokens = get_tokens(ground_truth)
        
        # Calculate overlap
        common = pred_tokens & gt_tokens
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = num_common / sum(pred_tokens.values())
        recall = num_common / sum(gt_tokens.values())
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def average_precision(
        retrieved_ids: List[Any],
        relevant_ids: List[Any],
    ) -> float:
        """
        Calculate Average Precision
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            
        Returns:
            Average Precision score
        """
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        score = 0.0
        num_hits = 0
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                num_hits += 1
                precision_at_i = num_hits / i
                score += precision_at_i
        
        return score / len(relevant_set)
    
    @staticmethod
    def mean_average_precision(
        retrieved_ids_list: List[List[Any]],
        relevant_ids_list: List[List[Any]],
    ) -> float:
        """
        Calculate Mean Average Precision (MAP)
        
        Args:
            retrieved_ids_list: List of retrieved document ID lists
            relevant_ids_list: List of relevant document ID lists
            
        Returns:
            MAP score
        """
        ap_scores = []
        
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            ap = EvaluationMetrics.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0


class LatencyMetrics:
    """Track and analyze latency metrics"""
    
    def __init__(self):
        self.latencies = []
    
    def add(self, latency_ms: float):
        """Add a latency measurement"""
        self.latencies.append(latency_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        return {
            'mean': np.mean(self.latencies),
            'median': np.median(self.latencies),
            'p95': np.percentile(self.latencies, 95),
            'p99': np.percentile(self.latencies, 99),
            'min': np.min(self.latencies),
            'max': np.max(self.latencies),
            'std': np.std(self.latencies),
        }
    
    def reset(self):
        """Reset latency measurements"""
        self.latencies = []


if __name__ == "__main__":
    # Example usage
    
    # Retrieval metrics
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc2', 'doc4', 'doc6']
    
    recall = EvaluationMetrics.recall_at_k(retrieved, relevant, k=5)
    precision = EvaluationMetrics.precision_at_k(retrieved, relevant, k=5)
    
    print(f"Recall@5: {recall:.4f}")
    print(f"Precision@5: {precision:.4f}")
    
    # QA metrics
    prediction = "Machine learning is a subset of AI"
    ground_truth = "Machine learning is a subset of artificial intelligence"
    
    em = EvaluationMetrics.exact_match(prediction, ground_truth)
    f1 = EvaluationMetrics.f1_score(prediction, ground_truth)
    
    print(f"\nExact Match: {em:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Latency metrics
    latency_tracker = LatencyMetrics()
    latency_tracker.add(150.5)
    latency_tracker.add(200.3)
    latency_tracker.add(175.8)
    
    stats = latency_tracker.get_stats()
    print(f"\nLatency Stats: {stats}")