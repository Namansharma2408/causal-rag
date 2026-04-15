"""
Thinking Mode Agent - Multi-Model Consensus System

This implements a sophisticated reasoning approach:
1. Run multiple models in parallel to answer the query
2. Each model anonymously reviews other models' answers
3. Aggregate feedback and select the best performer
4. Winner regenerates answer considering all feedback
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from ..models import Query, Document
from ..services.llmProvider import get_llm, UnifiedLLM, LLMProvider


class ModelRole(Enum):
    """Different model personas for diverse perspectives."""
    ANALYST = "analyst"      # Focuses on data patterns
    CRITIC = "critic"        # Skeptical, looks for gaps
    SYNTHESIZER = "synth"    # Connects ideas holistically  
    PRAGMATIST = "pragma"    # Focuses on actionable insights


@dataclass
class ModelResponse:
    """Response from a single model."""
    role: ModelRole
    answer: str
    confidence: float
    reasoning: str


@dataclass
class PeerReview:
    """Peer review of another model's answer."""
    reviewer_role: ModelRole
    target_role: ModelRole
    score: int  # 1-10
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]


@dataclass
class ThinkingResult:
    """Final result from thinking mode."""
    final_answer: str
    winning_model: ModelRole
    consensus_score: float
    all_responses: List[ModelResponse]
    all_reviews: List[PeerReview]
    refinement_notes: str
    metadata: Dict[str, Any]


class ThinkingAgent:
    """
    Multi-Model Consensus System for deep reasoning.
    
    Process:
    1. Four different models answer the query in parallel
    2. Each model reviews others' answers anonymously
    3. Scores are aggregated to find the best answer
    4. Winner refines their answer using all feedback
    """
    
    def __init__(self, llm: UnifiedLLM = None, models: List[str] = None):
        self.llm = llm or get_llm()
        # Use Ollama models for thinking mode - different models for diverse perspectives
        self.models = models or [
            "phi3:14b",
            "deepseek-r1:14b",
            "qwen2.5-coder:7b",
            "codellama:7b"
        ]
        self.max_workers = 2  # Parallel execution for Ollama
        
        # Map each model to a persona with different Ollama models
        self.model_personas = {
            ModelRole.ANALYST: "phi3:14b",
            ModelRole.CRITIC: "deepseek-r1:14b",
            ModelRole.SYNTHESIZER: "qwen2.5-coder:7b",
            ModelRole.PRAGMATIST: "codellama:7b",
        }
    
    def process(
        self, 
        query: Query, 
        documents: List[Document],
        extracted_info: str
    ) -> ThinkingResult:
        """
        Run the full thinking mode process.
        
        Args:
            query: The user's question
            documents: Retrieved documents
            extracted_info: Pre-extracted relevant information
            
        Returns:
            ThinkingResult with the refined consensus answer
        """
        # Step 1: Get parallel responses from all model personas
        responses = self._get_parallel_responses(query, extracted_info)
        
        # Step 2: Conduct anonymous peer reviews
        reviews = self._conduct_peer_reviews(query, responses)
        
        # Step 3: Aggregate scores and find winner
        winner, scores = self._aggregate_scores(responses, reviews)
        
        # Step 4: Generate refined answer with winner considering all feedback
        final_answer, refinement_notes = self._generate_refined_answer(
            query, extracted_info, winner, responses, reviews
        )
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(scores)
        
        return ThinkingResult(
            final_answer=final_answer,
            winning_model=winner.role,
            consensus_score=consensus_score,
            all_responses=responses,
            all_reviews=reviews,
            refinement_notes=refinement_notes,
            metadata={
                "scores": {r.role.value: s for r, s in zip(responses, scores)},
                "num_reviews": len(reviews),
                "models_used": len(responses)
            }
        )
    
    def _get_parallel_responses(
        self, 
        query: Query, 
        extracted_info: str
    ) -> List[ModelResponse]:
        """Get responses from all model personas in parallel."""
        
        personas = [
            (ModelRole.ANALYST, self._get_analyst_prompt),
            (ModelRole.CRITIC, self._get_critic_prompt),
            (ModelRole.SYNTHESIZER, self._get_synthesizer_prompt),
            (ModelRole.PRAGMATIST, self._get_pragmatist_prompt),
        ]
        
        responses = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for role, prompt_fn in personas:
                prompt = prompt_fn(query.text, extracted_info)
                future = executor.submit(self._get_model_response, role, prompt)
                futures[future] = role
            
            for future in as_completed(futures):
                role = futures[future]
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    # Create a fallback response on error
                    responses.append(ModelResponse(
                        role=role,
                        answer=f"Error generating response: {str(e)}",
                        confidence=0.0,
                        reasoning="Error occurred"
                    ))
        
        return responses
    
    def _get_model_response(self, role: ModelRole, prompt: str) -> ModelResponse:
        """Get response from a single model persona using its assigned model."""
        
        # Use the specific model assigned to this persona
        model_name = self.model_personas.get(role, "qwen2.5-coder:7b")
        
        # Generate using the specific model for this persona (Ollama for thinking mode)
        response = self.llm.generate(prompt, model=model_name, temperature=0.7, provider=LLMProvider.OLLAMA)
        
        # Parse the structured response
        answer = self._extract_section(response, "ANSWER")
        confidence = self._extract_confidence(response)
        reasoning = self._extract_section(response, "REASONING")
        
        return ModelResponse(
            role=role,
            answer=answer or response,
            confidence=confidence,
            reasoning=reasoning or "No explicit reasoning provided"
        )
    
    def _conduct_peer_reviews(
        self, 
        query: Query, 
        responses: List[ModelResponse]
    ) -> List[PeerReview]:
        """Each model reviews other models' answers anonymously."""
        
        reviews = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for reviewer in responses:
                for target in responses:
                    if reviewer.role != target.role:
                        future = executor.submit(
                            self._get_peer_review,
                            query.text,
                            reviewer.role,
                            target.role,
                            target.answer
                        )
                        futures.append(future)
            
            for future in as_completed(futures):
                try:
                    review = future.result()
                    if review:
                        reviews.append(review)
                except Exception:
                    pass
        
        return reviews
    
    def _get_peer_review(
        self, 
        question: str,
        reviewer_role: ModelRole,
        target_role: ModelRole,
        answer: str
    ) -> Optional[PeerReview]:
        """Get a single peer review."""
        
        prompt = f"""You are reviewing another AI's answer to a question.
Review objectively without knowing who wrote it.

QUESTION: {question}

ANSWER TO REVIEW:
{answer}

Provide your review in this exact format:

SCORE: [1-10, where 10 is excellent]

STRENGTHS:
- [strength 1]
- [strength 2]

WEAKNESSES:
- [weakness 1]
- [weakness 2]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]

Be specific and constructive."""

        response = self.llm.generate(prompt, model="codellama:7b", temperature=0.3, provider=LLMProvider.OLLAMA)
        
        # Parse review
        score = self._extract_score(response)
        strengths = self._extract_list(response, "STRENGTHS")
        weaknesses = self._extract_list(response, "WEAKNESSES")
        suggestions = self._extract_list(response, "SUGGESTIONS")
        
        return PeerReview(
            reviewer_role=reviewer_role,
            target_role=target_role,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions
        )
    
    def _aggregate_scores(
        self, 
        responses: List[ModelResponse],
        reviews: List[PeerReview]
    ) -> Tuple[ModelResponse, List[float]]:
        """Aggregate peer review scores to find the best response."""
        
        scores = {role: [] for role in ModelRole}
        
        # Collect all scores for each model
        for review in reviews:
            scores[review.target_role].append(review.score)
        
        # Calculate average scores
        avg_scores = []
        for response in responses:
            role_scores = scores[response.role]
            if role_scores:
                avg = sum(role_scores) / len(role_scores)
            else:
                avg = response.confidence * 10  # Fallback to self-confidence
            avg_scores.append(avg)
        
        # Find winner (highest average score)
        winner_idx = avg_scores.index(max(avg_scores))
        winner = responses[winner_idx]
        
        return winner, avg_scores
    
    def _generate_refined_answer(
        self,
        query: Query,
        extracted_info: str,
        winner: ModelResponse,
        all_responses: List[ModelResponse],
        all_reviews: List[PeerReview]
    ) -> Tuple[str, str]:
        """Winner generates refined answer considering all feedback."""
        
        # Compile all feedback for the winner
        winner_feedback = [r for r in all_reviews if r.target_role == winner.role]
        
        # Compile insights from other responses
        other_insights = []
        for resp in all_responses:
            if resp.role != winner.role:
                other_insights.append(f"- {resp.answer[:500]}...")
        
        # Compile all suggestions
        all_suggestions = []
        for review in all_reviews:
            all_suggestions.extend(review.suggestions)
        unique_suggestions = list(set(all_suggestions))[:10]
        
        # Compile all weaknesses to avoid
        all_weaknesses = []
        for review in all_reviews:
            all_weaknesses.extend(review.weaknesses)
        unique_weaknesses = list(set(all_weaknesses))[:10]
        
        prompt = f"""You provided an answer that was rated highest by peer reviewers.
Now refine your answer considering all the feedback and insights from other perspectives.

ORIGINAL QUESTION: {query.text}

YOUR ORIGINAL ANSWER:
{winner.answer}

PEER FEEDBACK ON YOUR ANSWER:
Strengths noted: {[s for r in winner_feedback for s in r.strengths]}
Weaknesses noted: {[w for r in winner_feedback for w in r.weaknesses]}

INSIGHTS FROM OTHER PERSPECTIVES:
{chr(10).join(other_insights[:3])}

ALL SUGGESTIONS TO CONSIDER:
{chr(10).join(['- ' + s for s in unique_suggestions])}

COMMON WEAKNESSES TO AVOID:
{chr(10).join(['- ' + w for w in unique_weaknesses])}

RELEVANT DATA:
{extracted_info[:2000]}

Now write your FINAL REFINED ANSWER that:
1. Keeps your original strengths
2. Addresses the weaknesses noted
3. Incorporates valuable insights from other perspectives
4. Follows the suggestions where appropriate
5. Avoids common pitfalls

Write a comprehensive, well-structured answer with clear sections.
Use markdown formatting for clarity."""

        # Use the winner's model for refinement instead of config reference (Ollama)
        refined_answer = self.llm.generate(prompt, model=self.model_personas[winner.role], temperature=0.4, provider=LLMProvider.OLLAMA)
        
        refinement_notes = f"""
Refinement based on {len(winner_feedback)} peer reviews.
Incorporated {len(unique_suggestions)} suggestions.
Addressed {len(unique_weaknesses)} common weaknesses.
Combined insights from {len(other_insights)} alternative perspectives.
"""
        
        return refined_answer, refinement_notes
    
    def _calculate_consensus(self, scores: List[float]) -> float:
        """Calculate consensus score (0-1) based on score variance."""
        if not scores:
            return 0.0
        
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        
        # Lower variance = higher consensus
        # Normalize: variance of 0 = 1.0 consensus, variance of 25 = 0.0
        consensus = max(0, 1 - (variance / 25))
        return round(consensus, 2)
    
    # === Persona Prompts ===
    
    def _get_analyst_prompt(self, question: str, context: str) -> str:
        """Analyst: Focuses on data patterns and statistics."""
        return f"""You are a Data Analyst AI. Your approach:
- Focus on patterns, trends, and quantifiable insights
- Look for correlations and statistical significance
- Cite specific examples and data points
- Be precise and evidence-based

QUESTION: {question}

CONTEXT DATA:
{context}

Provide your response in this format:

REASONING:
[Explain your analytical approach]

ANSWER:
[Your detailed, data-driven answer with specific examples]

CONFIDENCE: [0.0-1.0]"""

    def _get_critic_prompt(self, question: str, context: str) -> str:
        """Critic: Skeptical, looks for gaps and alternative explanations."""
        return f"""You are a Critical Analyst AI. Your approach:
- Question assumptions and look for gaps
- Consider alternative explanations
- Identify limitations and caveats
- Be thorough but fair in criticism

QUESTION: {question}

CONTEXT DATA:
{context}

Provide your response in this format:

REASONING:
[Explain your critical analysis approach]

ANSWER:
[Your balanced answer that addresses potential issues]

CONFIDENCE: [0.0-1.0]"""

    def _get_synthesizer_prompt(self, question: str, context: str) -> str:
        """Synthesizer: Connects ideas holistically."""
        return f"""You are a Synthesis AI. Your approach:
- Connect disparate pieces of information
- Find overarching themes and patterns
- Build comprehensive mental models
- See the big picture while respecting details

QUESTION: {question}

CONTEXT DATA:
{context}

Provide your response in this format:

REASONING:
[Explain your synthesis approach]

ANSWER:
[Your holistic, interconnected answer]

CONFIDENCE: [0.0-1.0]"""

    def _get_pragmatist_prompt(self, question: str, context: str) -> str:
        """Pragmatist: Focuses on actionable insights."""
        return f"""You are a Pragmatic Advisor AI. Your approach:
- Focus on practical, actionable insights
- Prioritize what matters most
- Give clear recommendations
- Consider implementation feasibility

QUESTION: {question}

CONTEXT DATA:
{context}

Provide your response in this format:

REASONING:
[Explain your practical approach]

ANSWER:
[Your actionable answer with clear recommendations]

CONFIDENCE: [0.0-1.0]"""

    # === Parsing Utilities ===
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from structured response."""
        pattern = rf'{section}:\s*\n?(.*?)(?=\n[A-Z]+:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from response."""
        pattern = r'CONFIDENCE:\s*([\d.]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5
    
    def _extract_score(self, text: str) -> int:
        """Extract review score."""
        pattern = r'SCORE:\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return min(10, max(1, int(match.group(1))))
            except ValueError:
                pass
        return 5
    
    def _extract_list(self, text: str, section: str) -> List[str]:
        """Extract a bulleted list from a section."""
        pattern = rf'{section}:\s*\n((?:[-•*]\s*.+\n?)+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            items = re.findall(r'[-•*]\s*(.+)', match.group(1))
            return [item.strip() for item in items if item.strip()]
        return []
