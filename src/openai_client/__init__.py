"""
OpenAI Client Python - A Python client for interacting with the OpenAI API.
"""

from .config.config import load
from .openai.api import OpenAIAPI
from .openai.models import (
    Message, ChatCompletionResponse, CompletionResponse, EmbeddingResponse,
    Model, ListModelsResponse, Assistant, Thread, ThreadMessage, Run,
    File, FineTuningJob
)

__version__ = "0.1.0"
__all__ = [
    "load",
    "OpenAIAPI",
    "Message",
    "ChatCompletionResponse",
    "CompletionResponse",
    "EmbeddingResponse",
    "Model",
    "ListModelsResponse",
    "Assistant",
    "Thread",
    "ThreadMessage",
    "Run",
    "File",
    "FineTuningJob"
] 