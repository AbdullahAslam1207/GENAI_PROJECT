"""
Prompt Templates Module
Contains various prompt templates for different tasks
"""

from typing import List, Dict, Any
from langchain.schema import Document


class PromptTemplates:
    """Collection of prompt templates for RAG tasks"""
    
    # Basic QA Template
    QA_TEMPLATE = """Answer the question based on the following context. If you cannot answer based on the context, say so clearly.
Cite the source numbers [1], [2], etc. when referencing specific information.

Context:
{context}

Question: {question}

Answer:"""

    # Conversational Template
    CONVERSATIONAL_TEMPLATE = """You are a helpful AI assistant. Use the following context to answer the user's question.
Be conversational, friendly, and cite sources when appropriate using [1], [2], etc.

Context:
{context}

Chat History:
{chat_history}

User: {question}
Assistant:"""

    # Detailed Analysis Template
    DETAILED_TEMPLATE = """You are an expert analyst. Provide a comprehensive answer to the question using the context below.
Include relevant details, cite sources [1], [2], etc., and organize your response clearly.

Context:
{context}

Question: {question}

Detailed Answer:"""

    # Summary Template
    SUMMARY_TEMPLATE = """Summarize the following context to answer the question.
Be concise but comprehensive, and cite key sources.

Context:
{context}

Question: {question}

Summary:"""

    # Factual Template (stricter)
    FACTUAL_TEMPLATE = """Answer the question using ONLY the information from the context below.
Do not add information from your general knowledge.
If the context doesn't contain the answer, say "The context does not contain this information."
Always cite sources [1], [2], etc.

Context:
{context}

Question: {question}

Factual Answer:"""

    # Multi-Document Synthesis Template
    SYNTHESIS_TEMPLATE = """Synthesize information from multiple sources to answer the question.
Compare and contrast different perspectives if they exist.
Cite all relevant sources [1], [2], etc.

Context from multiple sources:
{context}

Question: {question}

Synthesized Answer:"""

    # Explanatory Template
    EXPLANATORY_TEMPLATE = """Explain the answer to the question in a clear and educational manner.
Use the context provided and break down complex concepts.
Cite sources [1], [2], etc.

Context:
{context}

Question: {question}

Explanation:"""

    @staticmethod
    def format_context(documents: List[Document], max_length: int = 3000) -> str:
        """
        Format documents into context string
        
        Args:
            documents: List of documents
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            
            doc_text = f"[{i}] Source: {source}\n{content}\n\n"
            
            if total_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "".join(context_parts)
    
    @staticmethod
    def format_chat_history(chat_history: List[Dict[str, str]], max_turns: int = 3) -> str:
        """
        Format chat history into string
        
        Args:
            chat_history: List of chat turns
            max_turns: Maximum number of turns to include
            
        Returns:
            Formatted chat history
        """
        if not chat_history:
            return "No previous conversation."
        
        history_text = []
        for turn in chat_history[-max_turns:]:
            role = turn.get('role', 'user').capitalize()
            content = turn.get('content', '')
            history_text.append(f"{role}: {content}")
        
        return "\n".join(history_text)
    
    @staticmethod
    def create_custom_template(
        system_instruction: str,
        context_placeholder: str = "{context}",
        question_placeholder: str = "{question}",
        additional_fields: Dict[str, str] = None,
    ) -> str:
        """
        Create a custom prompt template
        
        Args:
            system_instruction: System-level instruction
            context_placeholder: Placeholder for context
            question_placeholder: Placeholder for question
            additional_fields: Additional field placeholders
            
        Returns:
            Custom template string
        """
        template = f"{system_instruction}\n\n"
        template += f"Context:\n{context_placeholder}\n\n"
        
        if additional_fields:
            for field_name, placeholder in additional_fields.items():
                template += f"{field_name}:\n{placeholder}\n\n"
        
        template += f"Question: {question_placeholder}\n\nAnswer:"
        
        return template


if __name__ == "__main__":
    # Example usage
    from langchain.schema import Document
    
    # Sample documents
    docs = [
        Document(page_content="ML is a subset of AI.", metadata={'source': 'doc1.pdf'}),
        Document(page_content="DL uses neural networks.", metadata={'source': 'doc2.pdf'}),
    ]
    
    # Format context
    context = PromptTemplates.format_context(docs)
    print("Formatted Context:")
    print(context)
    
    # Create custom template
    custom = PromptTemplates.create_custom_template(
        "You are a technical expert.",
        additional_fields={"Background": "{background}"}
    )
    print("\nCustom Template:")
    print(custom)