"""
Answer quality evaluation for CodeMind.

Tests answer accuracy and helpfulness with/without graph-based context.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class AnswerQuality(Enum):
    """Quality rating for answers."""
    EXCELLENT = 5  # Complete, accurate, well-explained
    GOOD = 4       # Mostly accurate, good explanation
    FAIR = 3       # Partially accurate, some issues
    POOR = 2       # Mostly inaccurate, unhelpful
    VERY_POOR = 1  # Completely wrong or irrelevant


@dataclass
class TestCase:
    """A test case for answer quality."""
    question: str
    expected_answer: str
    codebase_context: str
    query_type: str = "analysis"
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class AnswerQualityMetrics:
    """Metrics for answer quality."""
    accuracy: float  # 0-1, how correct the answer is
    completeness: float  # 0-1, how thoroughly it answers
    clarity: float  # 0-1, how well explained
    relevance: float  # 0-1, how relevant to the question
    hallucination_rate: float  # 0-1, frequency of false information
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (self.accuracy + self.completeness + self.clarity + self.relevance) / 4 * (1 - self.hallucination_rate)


@dataclass
class EvaluationResult:
    """Result of evaluating an answer."""
    test_case: TestCase
    answer: str
    metrics: AnswerQualityMetrics
    quality_rating: AnswerQuality
    with_context: bool
    latency_ms: float
    notes: str = ""


class AnswerEvaluator:
    """
    Evaluates answer quality by checking:
    - Correctness (does it match expected answer?)
    - Completeness (does it address all aspects?)
    - Clarity (is it well-explained?)
    - Hallucination (does it make up information?)
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[EvaluationResult] = []
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case for evaluation."""
        self.test_cases.append(test_case)
    
    def add_test_cases(self, test_cases: List[TestCase]) -> None:
        """Add multiple test cases."""
        self.test_cases.extend(test_cases)
    
    def evaluate_answer(
        self,
        answer: str,
        test_case: TestCase,
        with_context: bool = True,
        latency_ms: float = 0.0
    ) -> EvaluationResult:
        """
        Evaluate a single answer.
        
        Args:
            answer: The generated answer
            test_case: The test case
            with_context: Whether context was used
            latency_ms: Response latency
            
        Returns:
            EvaluationResult with metrics and rating
        """
        metrics = self._calculate_metrics(answer, test_case)
        rating = self._rate_quality(metrics)
        
        result = EvaluationResult(
            test_case=test_case,
            answer=answer,
            metrics=metrics,
            quality_rating=rating,
            with_context=with_context,
            latency_ms=latency_ms
        )
        
        self.results.append(result)
        return result
    
    def _calculate_metrics(self, answer: str, test_case: TestCase) -> AnswerQualityMetrics:
        """Calculate quality metrics for an answer."""
        # Note: These are simplified heuristics. A real implementation would use:
        # - BLEU/ROUGE scores for similarity
        # - Named entity matching for accuracy
        # - Semantic similarity for relevance
        # - Human evaluation for hallucination detection
        
        # Accuracy: How much of the expected answer is in the response
        expected_lower = test_case.expected_answer.lower()
        answer_lower = answer.lower()
        
        # Simple word overlap metric
        expected_words = set(expected_lower.split())
        answer_words = set(answer_lower.split())
        accuracy = len(expected_words & answer_words) / len(expected_words) if expected_words else 0
        
        # Completeness: Longer, more detailed answers are more complete
        completeness = min(len(answer) / 500, 1.0)  # Normalize to 500 chars
        
        # Clarity: Check for code examples, structure, punctuation
        has_code = '```' in answer or 'def ' in answer or 'class ' in answer
        has_structure = answer.count('\n') > 2
        clarity = 0.5 + (0.25 if has_code else 0) + (0.25 if has_structure else 0)
        
        # Relevance: How well the answer addresses the question
        question_words = set(test_case.question.lower().split())
        answer_words_set = set(answer_lower.split())
        relevance = len(question_words & answer_words_set) / len(question_words) if question_words else 0.5
        
        # Hallucination: Detect obvious false information
        # (In practice, would use semantic similarity to knowledge base)
        hallucination_rate = self._detect_hallucination(answer, test_case)
        
        return AnswerQualityMetrics(
            accuracy=accuracy,
            completeness=completeness,
            clarity=clarity,
            relevance=relevance,
            hallucination_rate=hallucination_rate
        )
    
    def _detect_hallucination(self, answer: str, test_case: TestCase) -> float:
        """
        Detect hallucinated information in answer.
        
        This is a simplified check. Real implementation would:
        - Verify facts against codebase
        - Check for impossible function names
        - Validate syntax examples
        """
        # Look for patterns that indicate hallucination
        hallucination_score = 0.0
        
        # If answer is very different from expected, likely hallucinated
        similarity = self._string_similarity(answer, test_case.expected_answer)
        if similarity < 0.2:
            hallucination_score += 0.3
        
        # Check for vague or generic answers that don't reference code
        if 'function' not in answer.lower() and 'code' not in answer.lower():
            hallucination_score += 0.1
        
        # If answer is suspiciously short, might be incomplete/hallucinated
        if len(answer) < 50:
            hallucination_score += 0.2
        
        return min(hallucination_score, 1.0)
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity (0-1)."""
        # Normalize
        s1 = str1.lower().split()
        s2 = str2.lower().split()
        
        # Word overlap
        overlap = len(set(s1) & set(s2))
        union = len(set(s1) | set(s2))
        
        return overlap / union if union > 0 else 0
    
    def _rate_quality(self, metrics: AnswerQualityMetrics) -> AnswerQuality:
        """Convert metrics to quality rating."""
        score = metrics.overall_score
        
        if score >= 0.85:
            return AnswerQuality.EXCELLENT
        elif score >= 0.70:
            return AnswerQuality.GOOD
        elif score >= 0.55:
            return AnswerQuality.FAIR
        elif score >= 0.40:
            return AnswerQuality.POOR
        else:
            return AnswerQuality.VERY_POOR
    
    def get_summary(self) -> Dict:
        """Get summary of evaluation results."""
        if not self.results:
            return {
                'total_tests': 0,
                'average_score': 0,
                'average_latency_ms': 0,
                'quality_distribution': {}
            }
        
        scores = [r.metrics.overall_score for r in self.results]
        latencies = [r.latency_ms for r in self.results]
        
        quality_counts = {}
        for quality in AnswerQuality:
            quality_counts[quality.name] = sum(1 for r in self.results if r.quality_rating == quality)
        
        return {
            'total_tests': len(self.results),
            'average_score': sum(scores) / len(scores),
            'average_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'quality_distribution': quality_counts,
            'results_with_context': sum(1 for r in self.results if r.with_context),
            'results_without_context': sum(1 for r in self.results if not r.with_context),
        }
    
    def compare_with_without_context(self) -> Dict:
        """Compare answer quality with vs without context."""
        with_context = [r for r in self.results if r.with_context]
        without_context = [r for r in self.results if not r.with_context]
        
        if not with_context or not without_context:
            return {'error': 'Need results with and without context for comparison'}
        
        with_scores = [r.metrics.overall_score for r in with_context]
        without_scores = [r.metrics.overall_score for r in without_context]
        
        with_latency = [r.latency_ms for r in with_context]
        without_latency = [r.latency_ms for r in without_context]
        
        return {
            'with_context': {
                'average_score': sum(with_scores) / len(with_scores),
                'average_latency_ms': sum(with_latency) / len(with_latency),
                'test_count': len(with_context)
            },
            'without_context': {
                'average_score': sum(without_scores) / len(without_scores),
                'average_latency_ms': sum(without_latency) / len(without_latency),
                'test_count': len(without_context)
            },
            'context_improvement': {
                'score_improvement': (sum(with_scores) / len(with_scores)) - (sum(without_scores) / len(without_scores)),
                'latency_overhead': (sum(with_latency) / len(with_latency)) - (sum(without_latency) / len(without_latency))
            }
        }
    
    def export_results(self, filepath: str) -> None:
        """Export evaluation results to JSON."""
        data = {
            'summary': self.get_summary(),
            'results': [
                {
                    'question': r.test_case.question,
                    'answer': r.answer,
                    'metrics': {
                        'accuracy': r.metrics.accuracy,
                        'completeness': r.metrics.completeness,
                        'clarity': r.metrics.clarity,
                        'relevance': r.metrics.relevance,
                        'overall_score': r.metrics.overall_score
                    },
                    'quality': r.quality_rating.name,
                    'with_context': r.with_context,
                    'latency_ms': r.latency_ms
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Golden set of test cases for evaluation
GOLDEN_TEST_SET = [
    TestCase(
        question="What does the authentication system do?",
        expected_answer="The authentication system verifies user credentials by checking username and password against stored hashes.",
        codebase_context="authentication",
        query_type="analysis",
        difficulty="easy"
    ),
    TestCase(
        question="How does the search feature work?",
        expected_answer="The search feature uses hybrid search combining vector embeddings for semantic similarity and BM25 for keyword matching.",
        codebase_context="search",
        query_type="analysis",
        difficulty="medium"
    ),
    TestCase(
        question="What is the call chain for the main function?",
        expected_answer="The main function initializes the app, sets up the database, and calls the route handlers.",
        codebase_context="main",
        query_type="architecture",
        difficulty="hard"
    ),
]
