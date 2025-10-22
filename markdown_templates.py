
## 🔧 **2. Markdown Template Engine**

"""
Markdown Template Engine for Prompt Engineering
Generates formatted prompts for any technique
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

class MarkdownTemplateEngine:
    """Generate beautiful Markdown-formatted prompts"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.techniques_metadata = self._load_techniques()
    
    def _load_techniques(self) -> Dict:
        """Load prompt technique metadata"""
        return {
            "zero-shot": {
                "name": "Zero-Shot Prompting",
                "emoji": "🎯",
                "complexity": "Low",
                "best_for": "Simple, direct questions",
                "tips": [
                    "Be specific and clear",
                    "State expected format",
                    "Avoid ambiguity"
                ]
            },
            "few-shot": {
                "name": "Few-Shot Prompting",
                "emoji": "📚",
                "complexity": "Low",
                "best_for": "Pattern matching, examples",
                "tips": [
                    "Provide 2-5 clear examples",
                    "Keep examples consistent",
                    "Show input → output format"
                ]
            },
            "cot": {
                "name": "Chain-of-Thought",
                "emoji": "🔗",
                "complexity": "Medium",
                "best_for": "Reasoning, analysis, math",
                "tips": [
                    "Add 'Let's think step-by-step'",
                    "Break into logical steps",
                    "Show reasoning process"
                ]
            },
            "tot": {
                "name": "Tree-of-Thoughts",
                "emoji": "🌳",
                "complexity": "High",
                "best_for": "Complex planning, alternatives",
                "tips": [
                    "Explore multiple paths",
                    "Evaluate each branch",
                    "Compare and select best"
                ]
            },
            "self-consistency": {
                "name": "Self-Consistency",
                "emoji": "🔄",
                "complexity": "High",
                "best_for": "Uncertain reasoning",
                "tips": [
                    "Generate multiple solutions",
                    "Vote on best answer",
                    "Increases reliability"
                ]
            },
            "generated-knowledge": {
                "name": "Generated Knowledge",
                "emoji": "📖",
                "complexity": "Medium",
                "best_for": "Fact-heavy tasks",
                "tips": [
                    "First generate relevant facts",
                    "Then apply to problem",
                    "Two-step process"
                ]
            },
            "toq": {
                "name": "Tree-of-Questions",
                "emoji": "❓",
                "complexity": "High",
                "best_for": "Deep research, investigation",
                "tips": [
                    "Decompose into sub-questions",
                    "Answer systematically",
                    "Build comprehensive view"
                ]
            },
            "react": {
                "name": "ReAct",
                "emoji": "⚡",
                "complexity": "High",
                "best_for": "Tool use, verification",
                "tips": [
                    "Alternate thought and action",
                    "Use for fact-checking",
                    "Integrate external tools"
                ]
            },
            "meta": {
                "name": "Meta Prompting",
                "emoji": "🔮",
                "complexity": "Medium",
                "best_for": "Prompt optimization",
                "tips": [
                    "Prompt about prompting",
                    "Self-improvement loop",
                    "Iterative refinement"
                ]
            },
            "role": {
                "name": "Role Prompting",
                "emoji": "🎭",
                "complexity": "Low",
                "best_for": "Expert perspectives",
                "tips": [
                    "Define clear expertise",
                    "Set context and tone",
                    "Maintain character"
                ]
            }
        }
    
    def _load_templates(self) -> Dict:
        """Load prompt transformation templates"""
        return {
            "zero-shot": """
{prompt}

Please provide a clear, direct answer.
""",
            
            "few-shot": """
Here are some examples of what I'm looking for:

Example 1:
Input: {example1_input}
Output: {example1_output}

Example 2:
Input: {example2_input}
Output: {example2_output}

Now, for my request:
{prompt}
""",
            
            "cot": """
{prompt}

Let's approach this step-by-step:
1. First, identify the key components
2. Then, analyze each part
3. Connect the insights
4. Formulate the complete answer

Think through each step carefully before proceeding.
""",
            
            "tot": """
{prompt}

Let's explore multiple approaches to this:

**Path A:** [Conservative approach]
- Pros:
- Cons:
- Outcome:

**Path B:** [Balanced approach]
- Pros:
- Cons:
- Outcome:

**Path C:** [Aggressive approach]
- Pros:
- Cons:
- Outcome:

Evaluate all paths and recommend the best one with justification.
""",
            
            "self-consistency": """
{prompt}

Generate 3 independent solutions to this problem:

**Solution 1:** [Complete reasoning]
**Solution 2:** [Different approach]
**Solution 3:** [Alternative method]

Now analyze all three and determine the most consistent/reliable answer.
""",
            
            "generated-knowledge": """
Step 1: Generate relevant knowledge about this topic:
{prompt}

First, list 5-10 important facts, principles, or concepts related to this query.

Step 2: Now apply this knowledge to answer the original question comprehensively.
""",
            
            "toq": """
Main Question: {prompt}

Let's break this down into sub-questions:

Q1: [Foundational question]
A1: 

Q2: [Context question]
A2:

Q3: [Detail question]
A3:

Q4: [Implication question]
A4:

Q5: [Synthesis question]
A5:

Final Comprehensive Answer:
[Synthesize all answers above]
""",
            
            "react": """
Question: {prompt}

Let me solve this using Reason + Act:

Thought 1: What do I need to know first?
Action 1: [Search/Calculate/Verify]
Observation 1: [Result]

Thought 2: Based on this, what's next?
Action 2: [Search/Calculate/Verify]
Observation 2: [Result]

Thought 3: How do I synthesize this?
Action 3: [Combine/Conclude]
Final Answer: [Complete solution]
""",
            
            "meta": """
Original Prompt: {prompt}

Let me optimize this prompt:

1. **Clarity Issues:** [Identify vague parts]
2. **Missing Context:** [What's needed]
3. **Output Format:** [Specify desired format]
4. **Constraints:** [Add boundaries]

**Optimized Prompt:**
[Enhanced version with all improvements]

**Why This Is Better:**
[Explanation of improvements]
""",
            
            "role": """
You are {role}, an expert in {domain} with {experience} years of experience.

Your expertise includes:
- {skill1}
- {skill2}
- {skill3}

Speaking from this perspective:
{prompt}

Provide insights that only someone with your expertise would know.
"""
        }
    
    def generate_prompt(self, 
                       original_prompt: str,
                       technique: str,
                       context: Optional[Dict] = None,
                       target_llm: str = "Universal") -> str:
        """
        Generate formatted prompt using specified technique
        
        Args:
            original_prompt: User's original prompt
            technique: Selected technique key
            context: Additional context/variables for template
            target_llm: Target LLM (for compatibility notes)
        
        Returns:
            Markdown-formatted enhanced prompt
        """
        
        if technique not in self.techniques_metadata:
            technique = "zero-shot"  # Fallback
        
        tech_info = self.techniques_metadata[technique]
        template = self.templates.get(technique, self.templates["zero-shot"])
        
        # Prepare context
        ctx = {"prompt": original_prompt}
        if context:
            ctx.update(context)
        
        # Fill template (basic - enhance as needed)
        if technique == "role" and not context:
            ctx.update({
                "role": "a senior expert",
                "domain": "the relevant field",
                "experience": "20+",
                "skill1": "Deep domain knowledge",
                "skill2": "Problem-solving",
                "skill3": "Clear communication"
            })
        elif technique == "few-shot" and not context:
            ctx.update({
                "example1_input": "[Your first example input]",
                "example1_output": "[Expected output]",
                "example2_input": "[Your second example input]",
                "example2_output": "[Expected output]"
            })
        
        # Format the enhanced prompt
        try:
            enhanced_prompt = template.format(**ctx)
        except KeyError:
            enhanced_prompt = template.replace("{prompt}", original_prompt)
        
        # Generate final Markdown output
        output = f"""# 🎯 Optimized Prompt

**Technique:** {tech_info['emoji']} {tech_info['name']}  
**Complexity:** {tech_info['complexity']}  
**Best For:** {tech_info['best_for']}  
**Target LLM:** {target_llm}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  

---

## 📝 Enhanced Prompt:

{enhanced_prompt}

---

## 💡 Usage Tips:

"""
        
        for tip in tech_info['tips']:
            output += f"- {tip}\n"
        
        output += f"""
---

## 🔄 Alternative Techniques to Try:

- If too complex → Try Zero-Shot
- If needs examples → Try Few-Shot  
- If needs reasoning → Try Chain-of-Thought
- If needs exploration → Try Tree-of-Thoughts

---

*Copy the enhanced prompt above and paste into your LLM*
"""
        
        return output
    
    def batch_generate(self, prompts: List[Dict]) -> List[str]:
        """Generate multiple prompts at once"""
        results = []
        for prompt_data in prompts:
            result = self.generate_prompt(
                prompt_data.get('prompt', ''),
                prompt_data.get('technique', 'zero-shot'),
                prompt_data.get('context'),
                prompt_data.get('target_llm', 'Universal')
            )
            results.append(result)
        return results
    
    def export_to_file(self, prompt: str, technique: str, 
                      filename: str = "optimized_prompt.md"):
        """Export generated prompt to Markdown file"""
        output = self.generate_prompt(prompt, technique)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output)
        return filename


# Example usage
if __name__ == "__main__":
    engine = MarkdownTemplateEngine()
    
    # Test 1: Simple optimization
    result = engine.generate_prompt(
        "How do I learn Python?",
        "cot"
    )
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Test 2: With context
    result = engine.generate_prompt(
        "Analyze this investment opportunity",
        "tot",
        context={
            "investment_type": "stocks",
            "amount": "$10,000",
            "timeline": "5 years"
        }
    )
    print(result)