"""
Decomposer agent - Analyzes query complexity and breaks down complex queries.
Only decomposes when necessary, keeping simple queries intact.
"""

from typing import Optional, Any, List
from dataclasses import dataclass

from .base import BaseAgent
from ..models import Query, AgentResponse
from ..services.llmProvider import get_llm, UnifiedLLM


@dataclass
class DecompositionResult:
    """Result of query decomposition analysis."""
    needs_decomposition: bool
    sub_queries: List[str]
    reasoning: str
    complexity_score: int  # 1-10 scale


class DecomposerAgent(BaseAgent):
    """
    Analyzes query complexity and decomposes complex queries into sub-queries.
    
    Key principle: Only decompose when necessary.
    - Simple queries (single concept, direct questions) → Pass through unchanged
    - Complex queries (multiple parts, comparisons, multi-step reasoning) → Decompose
    
    Complexity indicators:
    - Multiple questions in one (AND, also, as well as)
    - Comparison requests (compare, versus, difference between)
    - Multi-step reasoning (why does X cause Y which leads to Z)
    - Aggregation needs (all factors, complete list, everything about)
    """
    
    # Complexity threshold (1-10 scale, decompose if score >= threshold)
    COMPLEXITY_THRESHOLD = 7
    
    # Maximum sub-queries to prevent over-decomposition
    MAX_SUB_QUERIES = 4
    
    def __init__(self, llm: Optional[UnifiedLLM] = None):
        super().__init__("Decomposer")
        self.llm = llm or get_llm()
    
    def process(self, query: Query, context: Optional[Any] = None) -> AgentResponse:
        """
        Analyze query and decompose if complex.
        
        Returns:
            AgentResponse with DecompositionResult containing:
            - needs_decomposition: Whether query was decomposed
            - sub_queries: List of sub-queries (or original query if simple)
            - reasoning: Explanation of decision
            - complexity_score: 1-10 complexity rating
        """
        # First, analyze complexity
        complexity_result = self._analyze_complexity(query.text)
        
        if complexity_result.complexity_score < self.COMPLEXITY_THRESHOLD:
            # Simple query - pass through unchanged
            self.log(f"Simple query (score: {complexity_result.complexity_score}/10) - no decomposition needed")
            
            return AgentResponse(
                agent_name=self.name,
                result=DecompositionResult(
                    needs_decomposition=False,
                    sub_queries=[query.text],  # Original query as single item
                    reasoning=complexity_result.reasoning,
                    complexity_score=complexity_result.complexity_score
                ),
                metadata={
                    "decomposed": False,
                    "complexity_score": complexity_result.complexity_score
                }
            )
        
        # Complex query - decompose into sub-queries
        sub_queries = self._decompose_query(query.text)
        
        self.log(f"Complex query (score: {complexity_result.complexity_score}/10) - decomposed into {len(sub_queries)} sub-queries")
        
        return AgentResponse(
            agent_name=self.name,
            result=DecompositionResult(
                needs_decomposition=True,
                sub_queries=sub_queries,
                reasoning=complexity_result.reasoning,
                complexity_score=complexity_result.complexity_score
            ),
            metadata={
                "decomposed": True,
                "complexity_score": complexity_result.complexity_score,
                "sub_query_count": len(sub_queries)
            }
        )
    
    def _analyze_complexity(self, query_text: str) -> DecompositionResult:
        """
        Analyze query complexity to determine if decomposition is needed.
        
        Uses both heuristics and LLM analysis for robust detection.
        """
        # Quick heuristic check first (fast path for obviously simple queries)
        heuristic_score = self._heuristic_complexity_check(query_text)
        
        if heuristic_score <= 3:
            # Obviously simple - skip LLM call
            return DecompositionResult(
                needs_decomposition=False,
                sub_queries=[query_text],
                reasoning="Simple direct question - no decomposition needed",
                complexity_score=heuristic_score
            )
        
        # Use LLM for nuanced analysis
        prompt = f"""Analyze this query's complexity on a scale of 1-10.

Query: "{query_text}"

ONLY score HIGH (7-10) if the query has MULTIPLE of these:
- Multiple distinct questions combined (e.g., "What is X AND how does Y work?")
- Explicit comparison requests (e.g., "Compare X vs Y", "difference between")
- Multi-step causal chain reasoning (e.g., "Why does X cause Y which leads to Z?")
- Requests for comprehensive/exhaustive information about MULTIPLE topics

Score LOW (1-5) if the query:
- Asks about a single topic, even if complex
- Is a single "What causes..." or "Why does..." question
- Can be answered with information from one search

IMPORTANT: A single question about one topic should score 4-5, even if the topic is complex.
Only score 7+ if the query explicitly requires answering MULTIPLE separate questions.

Respond in this exact format:
COMPLEXITY_SCORE: [1-10]
REASONING: [One sentence explaining why]"""

        response = self.llm.generate_fast(prompt)
        
        # Parse response
        import re
        
        score = heuristic_score  # Fallback to heuristic
        reasoning = "Complexity analyzed"
        
        score_match = re.search(r'COMPLEXITY_SCORE:\s*(\d+)', response, re.IGNORECASE)
        if score_match:
            score = min(max(int(score_match.group(1)), 1), 10)
        
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return DecompositionResult(
            needs_decomposition=score >= self.COMPLEXITY_THRESHOLD,
            sub_queries=[query_text],
            reasoning=reasoning,
            complexity_score=score
        )
    
    def _heuristic_complexity_check(self, query_text: str) -> int:
        """
        Fast heuristic check for query complexity.
        Returns estimated complexity score 1-10.
        """
        query_lower = query_text.lower()
        word_count = len(query_text.split())
        score = 2  # Base score - assume simple by default
        
        # Check for multiple question marks (strong indicator)
        if query_text.count('?') > 1:
            score += 3
        
        # Check for comparison keywords (strong indicator)
        comparison_keywords = [
            'compare', 'versus', ' vs ', 'difference between',
            'differ', 'contrast', 'similarities', 'better than',
            'compared to'
        ]
        for keyword in comparison_keywords:
            if keyword in query_lower:
                score += 3
                break
        
        # Check for conjunction words indicating multiple parts
        multi_part_indicators = [
            ' and also ', ' as well as ', ' additionally ',
            ' plus ', ' along with ', ' together with '
        ]
        for indicator in multi_part_indicators:
            if indicator in query_lower:
                score += 2
                break
        
        # Simple ' and ' only adds +1 (very common in single questions)
        if ' and ' in query_lower and score < 5:
            score += 1
        
        # Check for comprehensive/exhaustive requests
        exhaustive_keywords = [
            'all factors', 'all reasons', 'everything about',
            'complete list', 'comprehensive', 'all the ways',
            'all possible', 'every aspect'
        ]
        for keyword in exhaustive_keywords:
            if keyword in query_lower:
                score += 2
                break
        
        # Check for causal chain keywords (multiple = complex)
        causal_keywords = [
            'leads to', 'results in', 'which causes', 'because of which',
            'chain', 'sequence of'
        ]
        causal_count = sum(1 for k in causal_keywords if k in query_lower)
        if causal_count >= 2:
            score += 2
        
        # Check query length (very long = might be complex)
        if word_count > 40:
            score += 2
        elif word_count > 25:
            score += 1
        
        # Simple query patterns (reduce score significantly)
        simple_starters = [
            'what is ', 'what are ', 'what causes ', 'what triggers ',
            'why do ', 'why does ', 'how do ', 'how does ',
            'explain ', 'describe ', 'define ', 'who is ', 'when did '
        ]
        for pattern in simple_starters:
            if query_lower.startswith(pattern):
                # Single topic question - cap the score
                if word_count < 15 and score > 4:
                    score = 4
                elif word_count < 20 and score > 5:
                    score = 5
                break
        
        return min(max(score, 1), 10)
    
    def _decompose_query(self, query_text: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        Each sub-query should be answerable independently.
        """
        prompt = f"""Break down this complex query into simpler, independent sub-questions.

Complex Query: "{query_text}"

Rules:
1. Create 2-4 sub-questions maximum
2. Each sub-question should be self-contained and answerable independently
3. Together, the sub-questions should cover all aspects of the original query
4. Keep sub-questions simple and focused on one concept each
5. Maintain the original intent and context

Format your response as:
SUB_QUERY_1: [first sub-question]
SUB_QUERY_2: [second sub-question]
SUB_QUERY_3: [third sub-question if needed]
SUB_QUERY_4: [fourth sub-question if needed]"""

        response = self.llm.generate_quality(prompt)
        
        # Parse sub-queries
        import re
        sub_queries = []
        
        for i in range(1, self.MAX_SUB_QUERIES + 1):
            match = re.search(rf'SUB_QUERY_{i}:\s*(.+?)(?=SUB_QUERY_|$)', response, re.IGNORECASE | re.DOTALL)
            if match:
                sub_query = match.group(1).strip()
                # Clean up the sub-query
                sub_query = sub_query.split('\n')[0].strip()  # Take first line only
                if sub_query and len(sub_query) > 10:
                    sub_queries.append(sub_query)
        
        # Fallback: if parsing failed, try line-by-line
        if not sub_queries:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering prefixes
                line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
                line = re.sub(r'^sub[_\-\s]*query[_\-\s]*\d*:?\s*', '', line, flags=re.IGNORECASE)
                if line and len(line) > 10 and '?' in line or len(line) > 20:
                    sub_queries.append(line)
                    if len(sub_queries) >= self.MAX_SUB_QUERIES:
                        break
        
        # Final fallback: return original query
        if not sub_queries:
            self.log("Decomposition parsing failed - using original query")
            return [query_text]
        
        return sub_queries[:self.MAX_SUB_QUERIES]
    
    def should_decompose(self, query_text: str) -> bool:
        """
        Quick check if a query needs decomposition.
        Useful for external callers who just need a yes/no answer.
        """
        query = Query(text=query_text)
        result = self.process(query)
        return result.result.needs_decomposition
