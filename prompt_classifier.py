"""
Intelligent Prompt Classifier with LlamaIndex Integration
Automatically selects the best prompt engineering technique
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# LlamaIndex imports
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

# Optional: For local models
# from llama_index.llms import Ollama
# from llama_index.embeddings import HuggingFaceEmbedding

import chromadb
from chromadb.config import Settings

# For PDF processing
import PyPDF2
import pdfplumber


@dataclass
class PromptTechnique:
    """Data class for prompt techniques"""
    name: str
    description: str
    best_for: List[str]
    triggers: List[str]
    complexity: str
    examples: List[str]
    template: str


class PromptClassifier:
    """
    Intelligent classifier that selects optimal prompt technique
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 use_local_model: bool = False,
                 chroma_path: str = "./chroma_db"):
        """
        Initialize the classifier

        Args:
            openai_api_key: OpenAI API key (optional if using local)
            use_local_model: Use Ollama/local models instead
            chroma_path: Path to ChromaDB storage
        """
        self.use_local = True  # Always use local models
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(exist_ok=True)
        
        # Initialize components
        self._setup_models()  # This will use local embeddings
        self._setup_vector_store()
        self._load_knowledge_base()
        # Always set up query engine after index creation/loading
        if hasattr(self, 'index') and self.index is not None:
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=3
            )
        else:
            self.query_engine = None
        # Technique patterns (rule-based backup)
        self.technique_patterns = self._init_patterns()
    
    def _setup_models(self):
        """Initialize LLM and embedding models"""
        # Always use local embeddings
        from llama_index.embeddings import HuggingFaceEmbedding
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        if self.use_local:
            # Use Ollama for local inference
            from llama_index.llms import Ollama
            self.llm = Ollama(model="llama2", temperature=0.1)
        else:
            # Use OpenAI
            self.llm = OpenAI(model="gpt-4", temperature=0.1)
    
    def _setup_vector_store(self):
        """Initialize ChromaDB vector store"""
        # Create Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name="prompt_techniques"
            )
        except:
            self.collection = self.chroma_client.get_collection(
                name="prompt_techniques"
            )
        
        # Create vector store with service context for local embeddings
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            llm=self.llm
        )
        
        # Create vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )
        
        # Set up storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create the index with local embeddings
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            service_context=service_context
        )
    
    def _init_patterns(self) -> Dict:
        """Initialize regex patterns for technique detection"""
        return {
            "cot": {
                "patterns": [
                    r"step.?by.?step",
                    r"explain.*how",
                    r"walk.*through",
                    r"reasoning",
                    r"why.*and.*how"
                ],
                "keywords": ["explain", "why", "how", "reason", "analyze"],
                "score_boost": 1.2
            },
            "tot": {
                "patterns": [
                    r"alternativ",
                    r"option",
                    r"different.*approach",
                    r"multiple.*path",
                    r"strateg"
                ],
                "keywords": ["alternatives", "options", "paths", "strategies", "compare"],
                "score_boost": 1.3
            },
            "few-shot": {
                "patterns": [
                    r"example",
                    r"like.*this",
                    r"similar.*to",
                    r"show.*me.*how",
                    r"demonstrat"
                ],
                "keywords": ["example", "like", "similar", "show", "demonstrate"],
                "score_boost": 1.1
            },
            "generated-knowledge": {
                "patterns": [
                    r"facts.*about",
                    r"research",
                    r"information.*on",
                    r"tell.*me.*about",
                    r"knowledge.*of"
                ],
                "keywords": ["facts", "research", "information", "knowledge", "data"],
                "score_boost": 1.15
            },
            "toq": {
                "patterns": [
                    r"investigat",
                    r"deep.*dive",
                    r"comprehensive",
                    r"all.*aspects",
                    r"thorough"
                ],
                "keywords": ["investigate", "comprehensive", "thorough", "explore", "all"],
                "score_boost": 1.25
            },
            "react": {
                "patterns": [
                    r"search",
                    r"verify",
                    r"check",
                    r"find.*out",
                    r"look.*up"
                ],
                "keywords": ["search", "verify", "check", "find", "lookup"],
                "score_boost": 1.2
            },
            "meta": {
                "patterns": [
                    r"improve.*prompt",
                    r"optimize.*prompt",
                    r"better.*way.*to.*ask",
                    r"rephrase",
                    r"rewrite"
                ],
                "keywords": ["improve", "optimize", "rephrase", "rewrite", "better"],
                "score_boost": 1.4
            },
            "role": {
                "patterns": [
                    r"act.*as",
                    r"pretend",
                    r"perspective.*of",
                    r"expert.*in",
                    r"as.*a"
                ],
                "keywords": ["act", "pretend", "perspective", "expert", "role"],
                "score_boost": 1.1
            },
            "self-consistency": {
                "patterns": [
                    r"most.*likely",
                    r"probable",
                    r"confident",
                    r"certain",
                    r"multiple.*answer"
                ],
                "keywords": ["likely", "probable", "confident", "certain", "consistent"],
                "score_boost": 1.3
            }
        }
    
    def _load_knowledge_base(self):
        """Load prompt engineering knowledge base"""
        # Check if documents are already in the collection
        if self.collection.count() > 0:
            return
            
        # Create knowledge base documents
        techniques_data = [
            {
                "name": "Chain-of-Thought",
                "content": """
                Chain-of-Thought (CoT) prompting enables step-by-step reasoning.
                Best for: Mathematical problems, logical reasoning, complex analysis.
                Triggers: 'step by step', 'explain how', 'walk through', 'why'.
                Example: 'Explain step by step how photosynthesis works.'
                Complexity: Medium
                """,
                "metadata": {"technique": "cot", "complexity": "medium"}
            },
            {
                "name": "Tree-of-Thoughts",
                "content": """
                Tree-of-Thoughts (ToT) explores multiple reasoning paths like a decision tree.
                Best for: Strategic planning, complex decision-making, creative problem-solving.
                Triggers: 'alternatives', 'options', 'strategies', 'different approaches'.
                Example: 'What are different strategies for launching a startup?'
                Complexity: High
                """,
                "metadata": {"technique": "tot", "complexity": "high"}
            },
            {
                "name": "Few-Shot Prompting",
                "content": """
                Few-Shot provides examples to guide the model's output format and style.
                Best for: Pattern matching, formatting, consistent outputs.
                Triggers: 'example', 'like this', 'similar to', 'show me'.
                Example: 'Give me examples like this: Input->Output'
                Complexity: Low
                """,
                "metadata": {"technique": "few-shot", "complexity": "low"}
            },
            {
                "name": "Generated Knowledge",
                "content": """
                Generated Knowledge first generates relevant facts, then applies them.
                Best for: Fact-heavy tasks, research questions, knowledge-intensive queries.
                Triggers: 'facts about', 'research on', 'information about'.
                Example: 'Generate facts about quantum computing, then explain its future.'
                Complexity: Medium
                """,
                "metadata": {"technique": "generated-knowledge", "complexity": "medium"}
            },
            {
                "name": "Tree-of-Questions",
                "content": """
                Tree-of-Questions breaks complex queries into sub-questions.
                Best for: Research, investigation, comprehensive analysis.
                Triggers: 'investigate', 'deep dive', 'all aspects', 'comprehensive'.
                Example: 'Investigate all aspects of climate change impacts.'
                Complexity: High
                """,
                "metadata": {"technique": "toq", "complexity": "high"}
            },
            {
                "name": "ReAct",
                "content": """
                ReAct combines reasoning with actions/tool use.
                Best for: Fact-checking, calculations, external tool integration.
                Triggers: 'search', 'verify', 'check', 'find out'.
                Example: 'Search and verify the latest statistics on renewable energy.'
                Complexity: High
                """,
                "metadata": {"technique": "react", "complexity": "high"}
            },
            {
                "name": "Meta Prompting",
                "content": """
                Meta Prompting optimizes the prompt itself.
                Best for: Prompt improvement, query refinement.
                Triggers: 'improve this prompt', 'better way to ask', 'optimize'.
                Example: 'Improve this prompt: How do I learn programming?'
                Complexity: Medium
                """,
                "metadata": {"technique": "meta", "complexity": "medium"}
            },
            {
                "name": "Role Prompting",
                "content": """
                Role Prompting assigns a specific role or expertise to the model.
                Best for: Expert perspectives, specialized knowledge.
                Triggers: 'act as', 'pretend you are', 'from perspective of'.
                Example: 'Act as a financial advisor and analyze this portfolio.'
                Complexity: Low
                """,
                "metadata": {"technique": "role", "complexity": "low"}
            },
            {
                "name": "Zero-Shot",
                "content": """
                Zero-Shot provides direct instructions without examples.
                Best for: Simple queries, straightforward tasks.
                Triggers: Basic questions without special requirements.
                Example: 'What is the capital of France?'
                Complexity: Low
                """,
                "metadata": {"technique": "zero-shot", "complexity": "low"}
            }
        ]
        
        # Convert to LlamaIndex documents
        documents = []
        for tech in techniques_data:
            doc = Document(
                text=tech["content"],
                metadata=tech["metadata"]
            )
            documents.append(doc)
        
        # Create service context with local embeddings
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            llm=self.llm
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=True
        )
        # Query engine will be set in __init__ after this method
    
    def classify_prompt(self, user_prompt: str) -> Tuple[str, float, str]:
        """
        Classify user prompt and return best technique
        
        Args:
            user_prompt: The user's input prompt
        
        Returns:
            Tuple of (technique_name, confidence_score, reasoning)
        """
        
        # Method 1: Pattern-based scoring
        pattern_scores = self._score_patterns(user_prompt)
        
        # Method 2: Semantic similarity search
        semantic_result = self._semantic_search(user_prompt)
        
        # Method 3: LLM-based classification
        llm_result = self._llm_classify(user_prompt)
        
        # Combine results (weighted voting)
        final_technique, confidence, reasoning = self._combine_results(
            pattern_scores, semantic_result, llm_result, user_prompt
        )
        
        return final_technique, confidence, reasoning
    
    def _score_patterns(self, prompt: str) -> Dict[str, float]:
        """Score prompt against pattern rules"""
        scores = {}
        prompt_lower = prompt.lower()
        
        for technique, rules in self.technique_patterns.items():
            score = 0.0
            
            # Check regex patterns
            for pattern in rules["patterns"]:
                if re.search(pattern, prompt_lower):
                    score += 1.0
            
            # Check keywords
            for keyword in rules["keywords"]:
                if keyword in prompt_lower:
                    score += 0.5
            
            # Apply boost
            score *= rules.get("score_boost", 1.0)
            
            scores[technique] = score
        
        # Add zero-shot as baseline
        if max(scores.values()) < 1.0:
            scores["zero-shot"] = 1.0
        
        return scores
    
    def _semantic_search(self, prompt: str) -> Dict:
        """Use vector similarity to find best technique"""
        response = self.query_engine.query(
            f"Which prompt engineering technique is best for: {prompt}"
        )
        
        # Extract technique from response metadata
        if response.source_nodes:
            best_node = response.source_nodes[0]
            technique = best_node.metadata.get("technique", "zero-shot")
            score = best_node.score if hasattr(best_node, 'score') else 0.5
            
            return {
                "technique": technique,
                "score": score,
                "context": str(response)[:200]
            }
        
        return {"technique": "zero-shot", "score": 0.3, "context": ""}
    
    def _llm_classify(self, prompt: str) -> Dict:
        """Use LLM to classify the prompt"""
        classification_prompt = f"""
        Analyze this user prompt and select the SINGLE best prompt engineering technique:
        
        User Prompt: "{prompt}"
        
        Available Techniques:
        - zero-shot: Direct, simple queries
        - few-shot: Needs examples
        - cot: Needs step-by-step reasoning
        - tot: Needs exploration of alternatives
        - generated-knowledge: Needs factual information first
        - toq: Needs comprehensive investigation
        - react: Needs tool use or verification
        - meta: Needs prompt optimization
        - role: Needs expert perspective
        - self-consistency: Needs high confidence through multiple attempts
        
        Return ONLY the technique name (e.g., 'cot'). No explanation.
        """
        
        response = self.llm.complete(classification_prompt)
        technique = str(response).strip().lower().replace("-", "")
        
        # Validate technique
        valid_techniques = ["zero-shot", "few-shot", "cot", "tot", 
                          "generated-knowledge", "toq", "react", 
                          "meta", "role", "self-consistency"]
        
        if technique not in [t.replace("-", "") for t in valid_techniques]:
            technique = "zero-shot"
        
        return {"technique": technique, "confidence": 0.8}
    
    def _combine_results(self, pattern_scores: Dict, semantic: Dict, 
                        llm: Dict, prompt: str) -> Tuple[str, float, str]:
        """Combine all classification methods"""
        
        # Normalize pattern scores
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 1
        normalized_patterns = {
            k: v/max_pattern_score for k, v in pattern_scores.items()
        }
        
        # Weight contributions
        weights = {
            "pattern": 0.3,
            "semantic": 0.3,
            "llm": 0.4
        }
        
        # Calculate final scores
        final_scores = {}
        
        # Add pattern scores
        for tech, score in normalized_patterns.items():
            final_scores[tech] = score * weights["pattern"]
        
        # Add semantic score
        sem_tech = semantic.get("technique", "zero-shot")
        if sem_tech in final_scores:
            final_scores[sem_tech] += semantic.get("score", 0.5) * weights["semantic"]
        else:
            final_scores[sem_tech] = semantic.get("score", 0.5) * weights["semantic"]
        
        # Add LLM score
        llm_tech = llm.get("technique", "zero-shot")
        if llm_tech in final_scores:
            final_scores[llm_tech] += llm.get("confidence", 0.8) * weights["llm"]
        else:
            final_scores[llm_tech] = llm.get("confidence", 0.8) * weights["llm"]
        
        # Get best technique
        best_technique = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_technique]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_technique, prompt, confidence)
        
        return best_technique, confidence, reasoning
    
    def _generate_reasoning(self, technique: str, prompt: str, confidence: float) -> str:
        """Generate explanation for technique selection"""
        
        reasons = {
            "cot": f"Requires step-by-step reasoning to break down the complexity",
            "tot": f"Involves exploring multiple strategies or alternatives",
            "few-shot": f"Benefits from examples to establish the pattern",
            "generated-knowledge": f"Needs factual information generation first",
            "toq": f"Requires comprehensive investigation through sub-questions",
            "react": f"Involves verification or tool use for accuracy",
            "meta": f"Focuses on optimizing the prompt itself",
            "role": f"Benefits from expert perspective or specific role",
            "self-consistency": f"Requires multiple attempts for confidence",
            "zero-shot": f"Simple enough for direct response"
        }
        
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        
        return (f"Selected {technique} (Confidence: {confidence_level}). "
                f"{reasons.get(technique, 'Best fit for this query type.')}")
    
    def process_and_enhance(self, user_prompt: str) -> Dict:
        """
        Complete pipeline: classify and enhance prompt
        
        Returns:
            Dict with technique, enhanced prompt, confidence, etc.
        """
        
        # Classify
        technique, confidence, reasoning = self.classify_prompt(user_prompt)
        
        # Import template engine
        from markdown_templates import MarkdownTemplateEngine
        engine = MarkdownTemplateEngine()
        
        # Generate enhanced prompt
        enhanced = engine.generate_prompt(
            user_prompt,
            technique.replace("-", "")
        )
        
        return {
            "original": user_prompt,
            "technique": technique,
            "confidence": confidence,
            "reasoning": reasoning,
            "enhanced_prompt": enhanced
        }


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = PromptClassifier(
        openai_api_key="your-api-key-here",  # Replace with your key
        use_local_model=False  # Set True for Ollama
    )
    
    # Test prompts
    test_prompts = [
        "How does photosynthesis work?",
        "Give me 3 examples of marketing strategies",
        "What are the alternatives for investing $10,000?",
        "Improve this prompt: How do I learn Python?",
        "Act as a chef and create a recipe",
        "Research everything about quantum computing",
        "What's 2+2?"
    ]
    
    for prompt in test_prompts:
        result = classifier.process_and_enhance(prompt)
        print(f"\n{'='*60}")
        print(f"Original: {prompt}")
        print(f"Technique: {result['technique']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Enhanced: [See Markdown output]")