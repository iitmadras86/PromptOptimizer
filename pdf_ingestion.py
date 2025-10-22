"""
PDF Ingestion System for Prompt Engineering Books
Converts PDFs to structured JSON knowledge base
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# PDF processing
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF for better extraction

# NLP for content analysis
import spacy
from sentence_transformers import SentenceTransformer

# For async processing
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptTechniqueEntry:
    """Structure for each technique extracted from PDFs"""
    technique_name: str
    source_book: str
    page_number: int
    description: str
    use_cases: List[str]
    examples: List[str]
    pros: List[str]
    cons: List[str]
    complexity: str
    keywords: List[str]
    related_techniques: List[str]
    
    def to_dict(self):
        return asdict(self)


class PDFKnowledgeExtractor:
    """
    Extract prompt engineering knowledge from PDF books
    """
    
    def __init__(self, output_dir: str = "./knowledge_base"):
        """
        Initialize the PDF extractor
        
        Args:
            output_dir: Directory to save extracted knowledge
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load NLP model for content analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Sentence transformer for similarity
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prompt technique patterns
        self.technique_patterns = self._init_technique_patterns()
        
        # Knowledge base
        self.knowledge_base = []
    
    def _init_technique_patterns(self) -> Dict:
        """Initialize patterns for detecting prompt techniques"""
        return {
            "patterns": {
                "zero-shot": [
                    r"zero[\s-]?shot",
                    r"direct prompting",
                    r"without examples"
                ],
                "few-shot": [
                    r"few[\s-]?shot",
                    r"with examples",
                    r"in[\s-]?context learning"
                ],
                "chain-of-thought": [
                    r"chain[\s-]?of[\s-]?thought",
                    r"CoT",
                    r"step[\s-]?by[\s-]?step reasoning"
                ],
                "tree-of-thoughts": [
                    r"tree[\s-]?of[\s-]?thoughts",
                    r"ToT",
                    r"reasoning paths",
                    r"thought branches"
                ],
                "self-consistency": [
                    r"self[\s-]?consistency",
                    r"multiple outputs",
                    r"voting mechanism"
                ],
                "react": [
                    r"ReAct",
                    r"reason[\s\+]act",
                    r"tool use",
                    r"action[\s-]?reasoning"
                ],
                "prompt-chaining": [
                    r"prompt[\s-]?chaining",
                    r"sequential prompts",
                    r"multi[\s-]?turn"
                ],
                "constitutional-ai": [
                    r"constitutional",
                    r"RLHF",
                    r"value alignment"
                ],
                "meta-prompting": [
                    r"meta[\s-]?prompt",
                    r"prompt about prompt",
                    r"self[\s-]?improvement"
                ]
            },
            "section_headers": [
                "technique",
                "method",
                "approach",
                "strategy",
                "pattern",
                "framework"
            ]
        }
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract structured knowledge from a single PDF
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of extracted technique entries
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []
        
        logger.info(f"Processing: {pdf_path.name}")
        
        extracted_data = []
        
        # Try multiple extraction methods for robustness
        try:
            # Method 1: PyMuPDF (best for complex layouts)
            extracted_data.extend(self._extract_with_pymupdf(pdf_path))
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        try:
            # Method 2: pdfplumber (good for tables)
            extracted_data.extend(self._extract_with_pdfplumber(pdf_path))
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Deduplicate and enhance
        extracted_data = self._deduplicate_entries(extracted_data)
        extracted_data = self._enhance_entries(extracted_data, pdf_path.name)
        
        return extracted_data
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Dict]:
        """Extract using PyMuPDF"""
        entries = []
        
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, 1):
                text = page.get_text()
                
                # Find technique mentions
                techniques_found = self._find_techniques_in_text(text)
                
                for technique in techniques_found:
                    # Extract context around technique
                    context = self._extract_context(text, technique, window=500)
                    
                    # Parse structured information
                    entry = self._parse_technique_info(
                        technique, context, pdf_path.name, page_num
                    )
                    
                    if entry:
                        entries.append(entry)
        
        return entries
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict]:
        """Extract using pdfplumber (better for tables)"""
        entries = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                
                # Check for tables (often contain comparisons)
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        entries.extend(
                            self._parse_table_for_techniques(
                                table, pdf_path.name, page_num
                            )
                        )
                
                # Regular text extraction
                techniques_found = self._find_techniques_in_text(text)
                for technique in techniques_found:
                    context = self._extract_context(text, technique, window=500)
                    entry = self._parse_technique_info(
                        technique, context, pdf_path.name, page_num
                    )
                    if entry:
                        entries.append(entry)
        
        return entries
    
    def _find_techniques_in_text(self, text: str) -> List[str]:
        """Find prompt technique mentions in text"""
        techniques_found = []
        text_lower = text.lower()
        
        for technique, patterns in self.technique_patterns["patterns"].items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    techniques_found.append(technique)
                    break
        
        return list(set(techniques_found))
    
    def _extract_context(self, text: str, technique: str, window: int = 500) -> str:
        """Extract context window around technique mention"""
        # Find all occurrences
        pattern = self.technique_patterns["patterns"][technique][0]
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return ""
        
        # Get context around first mention
        match = matches[0]
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        
        return text[start:end]
    
    def _parse_technique_info(self, technique: str, context: str, 
                             source: str, page: int) -> Optional[Dict]:
        """Parse structured information from context"""
        
        if not context:
            return None
        
        # Use NLP to extract information
        doc = self.nlp(context)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Categorize sentences
        description = []
        use_cases = []
        examples = []
        pros = []
        cons = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            # Classify sentence
            if any(word in sent_lower for word in ["is", "defined as", "refers to"]):
                description.append(sent)
            elif any(word in sent_lower for word in ["use when", "best for", "suitable"]):
                use_cases.append(sent)
            elif any(word in sent_lower for word in ["example", "for instance", "e.g."]):
                examples.append(sent)
            elif any(word in sent_lower for word in ["advantage", "benefit", "strength"]):
                pros.append(sent)
            elif any(word in sent_lower for word in ["disadvantage", "limitation", "drawback"]):
                cons.append(sent)
        
        # Extract keywords (nouns and verbs)
        keywords = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"] and not token.is_stop:
                keywords.append(token.lemma_)
        keywords = list(set(keywords))[:10]
        
        # Determine complexity
        complexity = self._assess_complexity(context)
        
        # Create entry
        entry = PromptTechniqueEntry(
            technique_name=technique.replace("-", " ").title(),
            source_book=source,
            page_number=page,
            description=" ".join(description[:2]) if description else "",
            use_cases=use_cases[:3],
            examples=examples[:2],
            pros=pros[:2],
            cons=cons[:2],
            complexity=complexity,
            keywords=keywords,
            related_techniques=[]  # Will be filled in enhancement
        )
        
        return entry.to_dict()
    
    def _parse_table_for_techniques(self, table: List[List], 
                                   source: str, page: int) -> List[Dict]:
        """Extract technique information from tables"""
        entries = []
        
        if not table or len(table) < 2:
            return entries
        
        # Assume first row is header
        headers = [str(h).lower() for h in table[0] if h]
        
        # Look for technique-related headers
        technique_col = None
        for i, header in enumerate(headers):
            if any(word in header for word in ["technique", "method", "approach"]):
                technique_col = i
                break
        
        if technique_col is None:
            return entries
        
        # Parse rows
        for row in table[1:]:
            if len(row) > technique_col and row[technique_col]:
                technique_name = str(row[technique_col])
                
                # Create basic entry from table data
                entry = {
                    "technique_name": technique_name,
                    "source_book": source,
                    "page_number": page,
                    "description": "",
                    "use_cases": [],
                    "examples": [],
                    "pros": [],
                    "cons": [],
                    "complexity": "Medium",
                    "keywords": [],
                    "related_techniques": []
                }
                
                # Try to extract more info from other columns
                for i, cell in enumerate(row):
                    if i != technique_col and cell:
                        header = headers[i] if i < len(headers) else ""
                        cell_text = str(cell)
                        
                        if "description" in header:
                            entry["description"] = cell_text
                        elif "use" in header or "when" in header:
                            entry["use_cases"] = [cell_text]
                        elif "example" in header:
                            entry["examples"] = [cell_text]
                        elif "pro" in header or "advantage" in header:
                            entry["pros"] = [cell_text]
                        elif "con" in header or "disadvantage" in header:
                            entry["cons"] = [cell_text]
                
                entries.append(entry)
        
        return entries
    
    def _assess_complexity(self, text: str) -> str:
        """Assess complexity level of technique"""
        complexity_indicators = {
            "low": ["simple", "basic", "straightforward", "easy"],
            "medium": ["moderate", "intermediate", "standard"],
            "high": ["complex", "advanced", "sophisticated", "difficult"]
        }
        
        text_lower = text.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(ind in text_lower for ind in indicators):
                return level.capitalize()
        
        return "Medium"  # Default
    
    def _deduplicate_entries(self, entries: List[Dict]) -> List[Dict]:
        """Remove duplicate entries based on similarity"""
        if len(entries) <= 1:
            return entries
        
        unique_entries = []
        seen_techniques = set()
        
        for entry in entries:
            key = f"{entry['technique_name']}_{entry['page_number']}"
            if key not in seen_techniques:
                seen_techniques.add(key)
                unique_entries.append(entry)
        
        return unique_entries
    
    def _enhance_entries(self, entries: List[Dict], source: str) -> List[Dict]:
        """Enhance entries with additional information"""
        
        # Find related techniques using embeddings
        if entries:
            descriptions = [e.get("description", "") for e in entries]
            if descriptions:
                embeddings = self.encoder.encode(descriptions)
                
                for i, entry in enumerate(entries):
                    # Find similar techniques
                    similarities = []
                    for j, other in enumerate(entries):
                        if i != j:
                            sim = self._cosine_similarity(
                                embeddings[i], embeddings[j]
                            )
                            if sim > 0.5:
                                similarities.append(other["technique_name"])
                    
                    entry["related_techniques"] = similarities[:3]
        
        return entries
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between vectors"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict:
        """
        Process multiple PDFs and create unified knowledge base
        
        Args:
            pdf_paths: List of PDF file paths
        
        Returns:
            Complete knowledge base dictionary
        """
        all_entries = []
        
        for pdf_path in pdf_paths:
            entries = self.extract_from_pdf(pdf_path)
            all_entries.extend(entries)
            logger.info(f"Extracted {len(entries)} entries from {Path(pdf_path).name}")
        
        # Create final knowledge base
        knowledge_base = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_entries": len(all_entries),
            "sources": [Path(p).name for p in pdf_paths],
            "techniques": all_entries,
            "metadata": {
                "unique_techniques": len(set(e["technique_name"] for e in all_entries)),
                "total_pages": sum(e["page_number"] for e in all_entries),
                "complexity_distribution": self._get_complexity_distribution(all_entries)
            }
        }
        
        # Save to JSON
        output_file = self.output_dir / "prompt_engineering_kb.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge base saved to: {output_file}")
        
        return knowledge_base
    
    def _get_complexity_distribution(self, entries: List[Dict]) -> Dict:
        """Get distribution of complexity levels"""
        distribution = {"Low": 0, "Medium": 0, "High": 0}
        for entry in entries:
            complexity = entry.get("complexity", "Medium")
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def create_llm_context_file(self, knowledge_base: Dict, 
                               output_file: str = "llm_context.md"):
        """
        Create a Markdown file optimized for LLM context
        
        Args:
            knowledge_base: The extracted knowledge base
            output_file: Output filename
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Prompt Engineering Knowledge Base\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Techniques Overview\n\n")
            
            # Group by technique name
            techniques_dict = {}
            for entry in knowledge_base["techniques"]:
                name = entry["technique_name"]
                if name not in techniques_dict:
                    techniques_dict[name] = []
                techniques_dict[name].append(entry)
            
            # Write each technique
            for technique_name, entries in techniques_dict.items():
                f.write(f"### {technique_name}\n\n")
                
                # Merge information from multiple sources
                all_descriptions = []
                all_use_cases = []
                all_examples = []
                all_pros = []
                all_cons = []
                
                for entry in entries:
                    if entry["description"]:
                        all_descriptions.append(entry["description"])
                    all_use_cases.extend(entry["use_cases"])
                    all_examples.extend(entry["examples"])
                    all_pros.extend(entry["pros"])
                    all_cons.extend(entry["cons"])
                
                # Write merged information
                if all_descriptions:
                    f.write(f"**Description:** {' '.join(all_descriptions[:2])}\n\n")
                
                if all_use_cases:
                    f.write("**Use Cases:**\n")
                    for uc in all_use_cases[:3]:
                        f.write(f"- {uc}\n")
                    f.write("\n")
                
                if all_examples:
                    f.write("**Examples:**\n")
                    for ex in all_examples[:2]:
                        f.write(f"- {ex}\n")
                    f.write("\n")
                
                if all_pros:
                    f.write("**Advantages:**\n")
                    for pro in all_pros[:2]:
                        f.write(f"- {pro}\n")
                    f.write("\n")
                
                if all_cons:
                    f.write("**Limitations:**\n")
                    for con in all_cons[:2]:
                        f.write(f"- {con}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        logger.info(f"LLM context file created: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = PDFKnowledgeExtractor(output_dir="./knowledge_base")
    
    # Example: Process your PDFs
    pdf_files = [
        "prompt_engineering_guide.pdf",
        "advanced_prompting_techniques.pdf",
        # Add your PDF paths here
    ]
    
    # Process PDFs (skip if files don't exist for demo)
    existing_pdfs = [pdf for pdf in pdf_files if Path(pdf).exists()]
    
    if existing_pdfs:
        # Extract knowledge
        kb = extractor.process_multiple_pdfs(existing_pdfs)
        
        # Create LLM-optimized context
        extractor.create_llm_context_file(kb)
        
        print(f"✅ Processed {len(existing_pdfs)} PDFs")
        print(f"📊 Extracted {kb['total_entries']} technique entries")
        print(f"💾 Knowledge base saved to ./knowledge_base/")
    else:
        # Create demo knowledge base
        demo_kb = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_entries": 9,
            "sources": ["demo"],
            "techniques": [
                {
                    "technique_name": "Chain of Thought",
                    "source_book": "Demo Book",
                    "page_number": 1,
                    "description": "Step-by-step reasoning approach",
                    "use_cases": ["Math problems", "Logic puzzles"],
                    "examples": ["Let's solve this step by step"],
                    "pros": ["Improves accuracy", "Shows reasoning"],
                    "cons": ["Longer responses", "More tokens"],
                    "complexity": "Medium",
                    "keywords": ["reasoning", "steps", "logic"],
                    "related_techniques": ["Tree of Thoughts"]
                }
            ],
            "metadata": {
                "unique_techniques": 1,
                "total_pages": 1,
                "complexity_distribution": {"Low": 0, "Medium": 1, "High": 0}
            }
        }
        
        # Save demo
        output_dir = Path("./knowledge_base")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "prompt_engineering_kb.json", 'w') as f:
            json.dump(demo_kb, f, indent=2)
        
        print("📝 Created demo knowledge base")
        print("Add your PDFs to the pdf_files list to process real data")