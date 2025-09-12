from typing import List

from src.retrieval.retriever import RetrievalResult


class PromptTemplates:
    @staticmethod
    def qa_prompt(context: str, question: str) -> str:
        """Generate Q&A prompt with context and citation requirements."""
        return f"""You are an expert FastAPI developer and documentation assistant. 
Your task is to answer questions about FastAPI based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question accurately based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific code examples when relevant
4. Cite sources by mentioning the relevant section or URL when available
5. Be concise but comprehensive
6. Use proper FastAPI terminology and best practices

ANSWER:"""

    @staticmethod
    def code_explanation_prompt(code: str, question: str) -> str:
        """Specialized prompt for code explanation and examples."""
        return f"""You are a FastAPI expert explaining code examples.

CODE EXAMPLE:
{code}

QUESTION: {question}

Please explain:
1. What this code does
2. Key FastAPI concepts demonstrated
3. Best practices shown or potential improvements
4. Common use cases for this pattern

EXPLANATION:"""

    @staticmethod
    def troubleshooting_prompt(context: str, error: str) -> str:
        """Debugging and troubleshooting assistance prompt."""
        return f"""You are a FastAPI troubleshooting expert.

CONTEXT FROM DOCUMENTATION:
{context}

ERROR/ISSUE: {error}

Please provide:
1. Most likely causes of this issue
2. Step-by-step troubleshooting guide
3. Code examples showing the correct approach
4. Prevention tips for the future

SOLUTION:"""

    @staticmethod
    def build_context_aware_prompt(question: str,
                                   results: List[RetrievalResult],
                                   prompt_type: str = "qa") -> str:
        """Build a context-aware prompt from retrieval results."""
        # Combine context from multiple sources
        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = f"Source {i}"
            if result.source_url:
                source_info += f" ({result.source_url})"

            context_part = f"{source_info}:\n{result.text}"
            context_parts.append(context_part)

        context = "\n\n---\n\n".join(context_parts)

        # Select appropriate prompt template
        if prompt_type == "qa":
            return PromptTemplates.qa_prompt(context, question)
        elif prompt_type == "code":
            return PromptTemplates.code_explanation_prompt(context, question)
        elif prompt_type == "troubleshooting":
            return PromptTemplates.troubleshooting_prompt(context, question)
        else:
            return PromptTemplates.qa_prompt(context, question)

    @staticmethod
    def system_prompt() -> str:
        """System prompt for consistent behavior."""
        return """You are FastAPI Assistant, an expert AI specialized in FastAPI 
framework documentation and best practices. You provide accurate, helpful, 
and practical answers about FastAPI development. Always base your responses 
on the provided documentation context and cite sources when possible."""
