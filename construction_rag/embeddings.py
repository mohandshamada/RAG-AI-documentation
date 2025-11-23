"""
Embeddings Handler
==================

Generates vector embeddings for text using various models.
Supports multiple embedding providers and models.
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    """
    Handles generation of vector embeddings for text.

    Supports:
    - Sentence Transformers (local, GPU-accelerated)
    - OpenAI embeddings (API)
    - Hugging Face models
    - Batch processing for efficiency
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding handler.

        Args:
            model_name: Name of the embedding model
            use_gpu: Whether to use GPU acceleration (for local models)
            api_key: API key for cloud providers (OpenAI, etc.)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.api_key = api_key
        self.model = None
        self.embedding_dimension = None

        # Detect provider and load model
        if model_name.startswith("openai/") or model_name.startswith("text-embedding"):
            self.provider = "openai"
            self._init_openai()
        elif model_name.startswith("sentence-transformers/") or "/" not in model_name:
            self.provider = "sentence-transformers"
            self._init_sentence_transformers()
        else:
            self.provider = "huggingface"
            self._init_huggingface()

        logger.info(f"Embedding handler initialized: {model_name} (dimension: {self.embedding_dimension})")

    def _init_sentence_transformers(self):
        """Initialize Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Remove prefix if present
            model_name = self.model_name.replace("sentence-transformers/", "")

            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

            logger.info(f"Loaded Sentence Transformer: {model_name} on {device}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            # Fallback to random embeddings
            self._init_fallback()

        except Exception as e:
            logger.error(f"Error loading Sentence Transformer: {str(e)}")
            self._init_fallback()

    def _init_openai(self):
        """Initialize OpenAI embeddings."""
        try:
            import openai

            if self.api_key:
                openai.api_key = self.api_key

            # Extract model name
            if self.model_name.startswith("openai/"):
                model = self.model_name.replace("openai/", "")
            else:
                model = self.model_name

            self.model = model

            # Set dimension based on model
            if "ada-002" in model:
                self.embedding_dimension = 1536
            elif "text-embedding-3-small" in model:
                self.embedding_dimension = 1536
            elif "text-embedding-3-large" in model:
                self.embedding_dimension = 3072
            else:
                self.embedding_dimension = 1536  # Default

            logger.info(f"OpenAI embeddings initialized: {model}")

        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            self._init_fallback()

    def _init_huggingface(self):
        """Initialize Hugging Face model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.to("cuda")

            # Get embedding dimension
            self.embedding_dimension = self.model.config.hidden_size

            logger.info(f"Loaded Hugging Face model: {self.model_name}")

        except ImportError:
            logger.error(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
            self._init_fallback()

        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {str(e)}")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback random embeddings."""
        logger.warning("Using fallback random embeddings (for testing only)")
        self.provider = "fallback"
        self.embedding_dimension = 384  # Default dimension
        self.model = None

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Single embedding or list of embeddings
        """
        # Handle single text
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        embeddings = []

        if self.provider == "sentence-transformers":
            embeddings = self._generate_sentence_transformers(texts, batch_size, show_progress)

        elif self.provider == "openai":
            embeddings = self._generate_openai(texts, batch_size)

        elif self.provider == "huggingface":
            embeddings = self._generate_huggingface(texts, batch_size)

        elif self.provider == "fallback":
            embeddings = self._generate_fallback(texts)

        # Return single embedding if single text was provided
        if single_text:
            return embeddings[0]

        return embeddings

    def _generate_sentence_transformers(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def _generate_openai(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        import openai

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = openai.Embedding.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                # Add zero vectors as fallback
                embeddings.extend([
                    [0.0] * self.embedding_dimension for _ in batch
                ])

        return embeddings

    def _generate_huggingface(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings using Hugging Face model."""
        import torch

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                if self.use_gpu and torch.cuda.is_available():
                    encoded = {k: v.to("cuda") for k, v in encoded.items()}

                # Get embeddings
                outputs = self.model(**encoded)

                # Use mean pooling
                batch_embeddings = self._mean_pooling(
                    outputs,
                    encoded['attention_mask']
                )

                embeddings.extend(batch_embeddings.cpu().numpy().tolist())

        return embeddings

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for Hugging Face models."""
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _generate_fallback(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings (fallback for testing)."""
        logger.warning("Generating random embeddings - not suitable for production")

        # Use hash of text for reproducibility
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.embedding_dimension).tolist()
            embeddings.append(embedding)

        return embeddings

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.embedding_dimension
