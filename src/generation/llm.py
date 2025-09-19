"""Language model interface and implementations optimized for macOS."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import subprocess
import json

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Abstract interface for language models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the model."""
        pass


class LocalLLM(LLMInterface):
    """Local language model using llama-cpp-python, optimized for Apple Silicon."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        use_metal: bool = True
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
            )

        if not model_path:
            model_path = self._find_or_download_model()

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {self.model_path}")

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=-1 if use_metal else 0,
            verbose=False
        )

        self.n_ctx = n_ctx
        self.model_name = self.model_path.name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using the local model."""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False
            )

            return response['choices'][0]['text'].strip()

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'context_length': self.n_ctx,
            'backend': 'llama.cpp with Metal' if self.llm.n_gpu_layers > 0 else 'llama.cpp CPU'
        }

    def _find_or_download_model(self) -> str:
        """Find or download a suitable model."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        common_models = [
            "llama-2-7b-chat.Q4_K_M.gguf",
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "phi-2.Q4_K_M.gguf"
        ]

        for model in common_models:
            model_path = models_dir / model
            if model_path.exists():
                return str(model_path)

        # Check if we should skip download (for demo purposes)
        import os
        if os.getenv('JOURNAL_SKIP_MODEL_DOWNLOAD'):
            raise FileNotFoundError("Model download disabled for demo")

        logger.info("No model found. Downloading Phi-2 (2.7B, works well on Mac)...")
        return self._download_phi2_model()

    def _download_phi2_model(self) -> str:
        """Download Phi-2 model from HuggingFace."""
        models_dir = Path("models")
        model_path = models_dir / "phi-2.Q4_K_M.gguf"

        if not model_path.exists():
            download_url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
            logger.info(f"Downloading model from {download_url}")

            try:
                import urllib.request
                urllib.request.urlretrieve(download_url, model_path)
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        return str(model_path)


class OllamaLLM(LLMInterface):
    """Language model using Ollama (must be installed separately)."""

    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self._check_ollama()
        self._ensure_model_pulled()

    def _check_ollama(self):
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama not found")
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not installed. "
                "Install from https://ollama.ai"
            )

    def _ensure_model_pulled(self):
        """Ensure the model is downloaded."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )

            if self.model_name not in result.stdout:
                logger.info(f"Pulling model {self.model_name}...")
                subprocess.run(
                    ["ollama", "pull", self.model_name],
                    check=True
                )
        except Exception as e:
            logger.warning(f"Could not check/pull model: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using Ollama."""
        try:
            cmd = [
                "ollama", "run",
                "--nowordwrap",
                self.model_name
            ]

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"Ollama error: {result.stderr}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error("Ollama generation timed out")
            return "Generation timed out"
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error generating response: {e}"

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'backend': 'Ollama',
            'api': 'subprocess'
        }


class MockLLM(LLMInterface):
    """Mock LLM for testing without a real model."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate mock response."""
        return (
            f"Based on your journal entries, I can see that this topic "
            f"appears frequently in your writing. The entries from recent months "
            f"show a pattern of reflection and growth around this subject. "
            f"Your earlier entries provide interesting context for understanding "
            f"how your thoughts have evolved over time."
        )

    def get_model_info(self) -> dict:
        """Get mock model info."""
        return {
            'model_name': 'mock',
            'backend': 'none',
            'note': 'This is a mock model for testing'
        }