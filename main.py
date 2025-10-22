# main.py - Complete working example

from markdown_templates import MarkdownTemplateEngine
from prompt_classifier import PromptClassifier
from pdf_ingestion import PDFKnowledgeExtractor
import os
from dotenv import load_dotenv

# Set your OpenAI API key
load_dotenv()  # loads .env from current working directory

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY not found in environment or .env")
os.environ["OPENAI_API_KEY"] = api_key


# 1. Ingest multiple PDF files to create knowledge base
pdf_files = [
    "Books/GooglePrompt.pdf",
    "Books/HubSpot.pdf"
]
extractor = PDFKnowledgeExtractor()
kb = extractor.process_multiple_pdfs(pdf_files)
print(f"Extracted {kb['total_entries']} entries from PDFs. Knowledge base saved to ./knowledge_base/prompt_engineering_kb.json")

# 2. Initialize classifier
classifier = PromptClassifier()

# 3. Test the system
user_prompt = "What are the best strategies for investing $10,000?"

# Get enhanced prompt
result = classifier.process_and_enhance(user_prompt)

print("Original:", result["original"])
print("Technique:", result["technique"])
print("Confidence:", result["confidence"])
print("\nEnhanced Prompt:\n")
print(result["enhanced_prompt"])

# 4. Save to file for copy-paste