"""
LLM Wrapper Module
Handles interaction with LLM (Groq API with Llama 3.1)
"""

import os
from typing import List, Dict, Any, Optional
import logging

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMWrapper:
    """Wrapper for LLM interaction using Groq API"""
    
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: str = None,
    ):
        """
        Initialize LLM wrapper
        
        Args:
            model_name: Name of the model
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            api_key: Groq API key (uses env variable if None)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API key
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.api_key = api_key
        
        # Initialize LLM
        logger.info(f"Initializing Groq LLM with model: {model_name}")
        self.llm = self._initialize_llm(model_name)
        logger.info("LLM initialized successfully")
    
    def _initialize_llm(self, model_name: str):
        """
        Initialize LLM with specified model
        
        Args:
            model_name: Name of the model
            
        Returns:
            ChatGroq instance
        """
        return ChatGroq(
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            groq_api_key=self.api_key,
        )
    
    def generate(
        self,
        prompt: str,
        system_message: str = None,
    ) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_with_context(
        self,
        query: str,
        context_documents: List[Document],
        template: str = None,
        max_context_length: int = 3000,
    ) -> str:
        """
        Generate response with retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved documents
            template: Custom prompt template
            max_context_length: Maximum context length in characters
            
        Returns:
            Generated answer
        """
        # Format context from documents
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(context_documents, 1):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            
            # Check if adding this document exceeds max length
            doc_text = f"[{i}] Source: {source}\n{content}\n\n"
            if total_length + len(doc_text) > max_context_length:
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        context = "".join(context_parts)
        
        # Use default template if none provided
        if template is None:
            template = """Answer the question based on the following context. If you cannot answer based on the context, say so.
Cite the source numbers [1], [2], etc. when referencing specific information.

Context:
{context}

Question: {question}

Answer:"""
        
        # Format prompt
        prompt = template.format(context=context, question=query)
        
        # Generate response
        response = self.generate(prompt)
        
        return response
    
    def generate_conversational(
        self,
        query: str,
        context_documents: List[Document],
        chat_history: List[Dict[str, str]] = None,
        template: str = None,
    ) -> str:
        """
        Generate conversational response with history
        
        Args:
            query: User query
            context_documents: Retrieved documents
            chat_history: Previous conversation history
            template: Custom prompt template
            
        Returns:
            Generated answer
        """
        # Format context
        context_parts = []
        for i, doc in enumerate(context_documents[:5], 1):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        # Format chat history
        history_text = ""
        if chat_history:
            for turn in chat_history[-3:]:  # Last 3 turns
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                history_text += f"{role.capitalize()}: {content}\n"
        
        # Use default template if none provided
        if template is None:
            template = """You are a helpful assistant. Use the following context to answer the user's question.
Be conversational and cite sources when appropriate.

Context:
{context}

Chat History:
{chat_history}

User: {question}
Assistant:"""
        
        # Format prompt
        prompt = template.format(
            context=context,
            chat_history=history_text,
            question=query
        )
        
        # Generate response
        response = self.generate(prompt)
        
        return response
    
    def batch_generate(
        self,
        queries: List[str],
        contexts: List[List[Document]],
        template: str = None,
    ) -> List[str]:
        """
        Generate responses for multiple queries
        
        Args:
            queries: List of queries
            contexts: List of context document lists
            template: Custom prompt template
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for query, context_docs in zip(queries, contexts):
            response = self.generate_with_context(
                query,
                context_docs,
                template=template
            )
            responses.append(response)
        
        return responses
    
    def stream_generate(
        self,
        query: str,
        context_documents: List[Document],
        template: str = None,
    ):
        """
        Stream generation (for real-time display)
        
        Args:
            query: User query
            context_documents: Retrieved documents
            template: Custom prompt template
            
        Yields:
            Generated text chunks
        """
        # Format context
        context_parts = []
        for i, doc in enumerate(context_documents[:5], 1):
            content = doc.page_content
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        # Use default template if none provided
        if template is None:
            template = """Answer the question based on the following context. Cite sources [1], [2], etc.

Context:
{context}

Question: {question}

Answer:"""
        
        # Format prompt
        prompt = template.format(context=context, question=query)
        
        # Stream response
        messages = [HumanMessage(content=prompt)]
        
        try:
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample context documents
    documents = [
        Document(
            page_content="Machine learning is a subset of AI that enables systems to learn from data.",
            metadata={'source': 'ml_intro.pdf'}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={'source': 'dl_basics.pdf'}
        ),
    ]
    
    # Initialize LLM
    try:
        llm = LLMWrapper()
        
        # Generate response
        query = "What is machine learning?"
        response = llm.generate_with_context(query, documents)
        
        print(f"Query: {query}")
        print(f"Response: {response}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set GROQ_API_KEY in your .env file")