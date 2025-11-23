"""
Query Engine for Construction RAG
==================================

Handles user queries and generates responses using retrieved context.
Integrates with LLMs for answer generation.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine for RAG system.

    Features:
    - Semantic search using vector store
    - Context assembly from retrieved chunks
    - LLM integration for answer generation
    - Construction-specific prompting
    - Source attribution
    """

    def __init__(
        self,
        vector_store,
        embedding_handler,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4",
        llm_provider: str = "openai"
    ):
        """
        Initialize query engine.

        Args:
            vector_store: VectorStore instance
            embedding_handler: EmbeddingHandler instance
            llm_api_key: API key for LLM service
            llm_model: LLM model to use
            llm_provider: LLM provider (openai, anthropic, etc.)
        """
        self.vector_store = vector_store
        self.embedding_handler = embedding_handler
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_provider = llm_provider

        # Initialize LLM client
        self.llm_available = False
        if llm_api_key:
            self._init_llm()

    def _init_llm(self):
        """Initialize LLM client."""
        if self.llm_provider == "openai":
            try:
                import openai
                openai.api_key = self.llm_api_key
                self.llm_available = True
                logger.info(f"LLM initialized: {self.llm_model}")
            except ImportError:
                logger.error("openai not installed. Install with: pip install openai")
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.llm_api_key)
                self.llm_available = True
                logger.info(f"LLM initialized: {self.llm_model}")
            except ImportError:
                logger.error("anthropic not installed. Install with: pip install anthropic")
        else:
            logger.warning(f"Unsupported LLM provider: {self.llm_provider}")

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_sources: bool = True,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Process a query and generate an answer.

        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            filter_metadata: Filter results by metadata
            return_sources: Whether to include source information
            temperature: LLM temperature (0 = deterministic, 1 = creative)

        Returns:
            Dictionary with answer and optional sources
        """
        # Generate query embedding
        query_embedding = self.embedding_handler.generate_embeddings([question])[0]

        # Retrieve relevant chunks
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        if not results:
            return {
                "answer": "I couldn't find any relevant information in the construction documents to answer your question.",
                "sources": [],
                "context_used": False
            }

        # Assemble context
        context = self._assemble_context(results)

        # Generate answer
        if self.llm_available:
            answer = self._generate_answer_with_llm(question, context, temperature)
        else:
            answer = self._generate_answer_without_llm(question, results)

        # Prepare response
        response = {
            "answer": answer,
            "context_used": True,
            "num_sources": len(results)
        }

        if return_sources:
            response["sources"] = self._format_sources(results)

        return response

    def _assemble_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Assemble context from retrieved chunks.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            text = result.get('text', '')

            # Add source information
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'N/A')

            context_parts.append(
                f"[Source {i}: {file_name}, Page {page}]\n{text}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_answer_with_llm(
        self,
        question: str,
        context: str,
        temperature: float
    ) -> str:
        """
        Generate answer using LLM.

        Args:
            question: User's question
            context: Retrieved context
            temperature: LLM temperature

        Returns:
            Generated answer
        """
        # Construct prompt
        prompt = self._construct_prompt(question, context)

        try:
            if self.llm_provider == "openai":
                import openai

                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant specialized in construction engineering. "
                                       "Answer questions based on the provided context from construction documents. "
                                       "If the context doesn't contain relevant information, say so. "
                                       "Always cite specific sources when providing information."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )

                return response.choices[0].message.content

            elif self.llm_provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model=self.llm_model,
                    max_tokens=1000,
                    temperature=temperature,
                    system="You are a helpful assistant specialized in construction engineering. "
                           "Answer questions based on the provided context from construction documents. "
                           "If the context doesn't contain relevant information, say so. "
                           "Always cite specific sources when providing information.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _generate_answer_without_llm(
        self,
        question: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer without LLM (fallback mode).

        Args:
            question: User's question
            results: Search results

        Returns:
            Simple answer based on retrieved chunks
        """
        answer_parts = [
            "Based on the construction documents, here are the most relevant excerpts:\n"
        ]

        for i, result in enumerate(results[:3], 1):
            metadata = result.get('metadata', {})
            text = result.get('text', '')[:500]  # Limit length

            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'N/A')

            answer_parts.append(
                f"\n{i}. From {file_name} (Page {page}):\n{text}...\n"
            )

        answer_parts.append(
            "\nNote: LLM not configured. Install an LLM API and provide API key for better answers."
        )

        return "".join(answer_parts)

    def _construct_prompt(self, question: str, context: str) -> str:
        """
        Construct prompt for LLM.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following context from construction documents, please answer the question.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific sources (e.g., "According to [Source X]...")
- If the context doesn't contain enough information, acknowledge this
- For technical specifications, include exact values and requirements
- For drawings or IFC data, describe relevant elements and properties
- Be concise but comprehensive

Answer:"""

        return prompt

    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source information for response.

        Args:
            results: Search results

        Returns:
            List of formatted source dictionaries
        """
        sources = []

        for result in results:
            metadata = result.get('metadata', {})

            source = {
                "file_name": metadata.get('file_name', 'Unknown'),
                "file_type": metadata.get('file_type', 'Unknown'),
                "document_type": metadata.get('document_type', 'Unknown'),
                "page": metadata.get('page'),
                "chunk_type": metadata.get('chunk_type'),
                "text_preview": result.get('text', '')[:200] + "...",
                "relevance_score": result.get('score')
            }

            # Add IFC-specific metadata if available
            if metadata.get('element_type'):
                source['element_type'] = metadata.get('element_type')
                source['element_count'] = metadata.get('element_count')

            sources.append(source)

        return sources

    def batch_query(
        self,
        questions: List[str],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.

        Args:
            questions: List of questions
            top_k: Number of relevant chunks per query
            filter_metadata: Filter results by metadata

        Returns:
            List of responses
        """
        responses = []

        for question in questions:
            response = self.query(
                question=question,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            responses.append(response)

        return responses
