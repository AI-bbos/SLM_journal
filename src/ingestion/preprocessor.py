"""Text preprocessing and chunking for journal entries."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    max_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_size: int = 100
    preserve_sentences: bool = True


class TextPreprocessor:
    """Preprocessor for cleaning and chunking journal text."""

    def __init__(self, chunk_config: Optional[ChunkConfig] = None):
        self.config = chunk_config or ChunkConfig()
        self.sentence_endings = re.compile(r'[.!?]\s+')

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '--').replace('…', '...')

        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.

        Returns:
            List of tuples (chunk_text, start_idx, end_idx)
        """
        if not text:
            return []

        text = self.clean_text(text)
        words = text.split()

        if len(words) <= self.config.max_tokens:
            return [(text, 0, len(words))]

        chunks = []
        start_idx = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.config.max_tokens, len(words))

            if self.config.preserve_sentences and end_idx < len(words):
                chunk_text = ' '.join(words[start_idx:end_idx])
                last_period = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )

                if last_period > self.config.min_chunk_size:
                    actual_words = chunk_text[:last_period + 1].split()
                    end_idx = start_idx + len(actual_words)

            chunk = ' '.join(words[start_idx:end_idx])
            chunks.append((chunk, start_idx, end_idx))

            if end_idx >= len(words):
                break

            start_idx = end_idx - self.config.overlap_tokens

        return chunks

    def extract_metadata(self, text: str) -> dict:
        """Extract metadata from text content."""
        metadata = {}

        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            metadata['urls'] = list(set(urls))

        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata['emails'] = list(set(emails))

        questions = re.findall(r'[^.!?]*\?', text)
        if questions:
            metadata['questions'] = [q.strip() for q in questions[:5]]

        metadata['word_count'] = len(text.split())
        metadata['sentence_count'] = len(self.sentence_endings.split(text))

        emotions = {
            'happy': r'\b(happy|joy|excited|wonderful|great|amazing|fantastic|excellent)\b',
            'sad': r'\b(sad|depressed|unhappy|miserable|terrible|awful|horrible)\b',
            'anxious': r'\b(anxious|worried|nervous|stressed|tense|overwhelmed)\b',
            'grateful': r'\b(grateful|thankful|appreciative|blessed|fortunate)\b',
            'angry': r'\b(angry|mad|furious|frustrated|irritated|annoyed)\b'
        }

        detected_emotions = []
        for emotion, pattern in emotions.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_emotions.append(emotion)

        if detected_emotions:
            metadata['emotions'] = detected_emotions

        return metadata

    def generate_chunk_id(self, content: str, file_path: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk."""
        id_string = f"{file_path}:{chunk_index}:{content[:50]}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]