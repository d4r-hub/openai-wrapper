from typing import List, Optional, Dict, Any, Union
from .client import Client
from .models import (
    Message, ChatCompletionResponse, CompletionResponse, EmbeddingResponse,
    Model, ListModelsResponse, Assistant, Thread, ThreadMessage, Run,
    File, FineTuningJob, AssistantTool, AssistantFile, AssistantListResponse,
    ThreadListResponse, MessageListResponse, RunListResponse, FileListResponse,
    FineTuningJobListResponse, CreateAssistantRequest, CreateThreadRequest,
    CreateMessageRequest, CreateRunRequest, CreateFineTuningJobRequest,
    ErrorResponse, ModerationResponse, ImageGenerationRequest, ImageGenerationResponse,
    ImageEditRequest, ImageVariationRequest, TranscriptionRequest, TranscriptionResponse,
    TranslationRequest, TranslationResponse
)

class OpenAIAPI(Client):
    def chat_completion(
        self,
        messages: List[Message],
        model: str = "gpt-4-turbo",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        store: Optional[bool] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """Create a chat completion."""
        data = {
            "model": model,
            "messages": [msg.dict() for msg in messages],
            **kwargs
        }
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if store is not None:
            data["store"] = store

        response = self._post("chat/completions", json=data)
        return ChatCompletionResponse(**response)

    def completion(
        self,
        prompt: str,
        model: str = "text-davinci-003",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Create a completion."""
        data = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        response = self._post("completions", json=data)
        return CompletionResponse(**response)

    def create_embeddings(
        self,
        input: List[str],
        model: str = "text-embedding-ada-002",
        **kwargs
    ) -> EmbeddingResponse:
        """Create embeddings for a list of inputs."""
        data = {
            "model": model,
            "input": input,
            **kwargs
        }
        response = self._post("embeddings", json=data)
        return EmbeddingResponse(**response)

    def list_models(self) -> ListModelsResponse:
        """List all available models."""
        response = self._get("models")
        return ListModelsResponse(**response)

    def retrieve_model(self, model_id: str) -> Model:
        """Retrieve a specific model."""
        response = self._get(f"models/{model_id}")
        return Model(**response)

    # Assistants API
    def get_assistant_by_name(self, name: str) -> Optional[Assistant]:
        """Get an assistant by name if it exists."""
        assistants = self.list_assistants()
        for assistant in assistants.data:
            if assistant.name == name:
                return assistant
        return None

    def create_assistant(
        self,
        name: str,
        model: str,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_existing: bool = True
    ) -> Assistant:
        """Create a new assistant.
        
        Args:
            name: The name of the assistant
            model: The model to use for the assistant
            description: Optional description of the assistant
            instructions: Optional instructions for the assistant
            tools: Optional list of tools the assistant can use
            metadata: Optional metadata for the assistant
            skip_existing: If True, return existing assistant if one with same name exists
            
        Returns:
            Assistant: The created or existing assistant
            
        Raises:
            ValueError: If skip_existing is False and an assistant with the same name exists
        """
        if skip_existing:
            existing = self.get_assistant_by_name(name)
            if existing:
                return existing

        data = {
            "name": name,
            "model": model,
            "description": description,
            "instructions": instructions,
            "tools": tools or [],
            "metadata": metadata
        }
        response = self._post("assistants", json=data)
        return Assistant(**response)

    def list_assistants(
        self,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> AssistantListResponse:
        """List all assistants with pagination."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get("assistants", params=params)
        return AssistantListResponse(**response)

    def retrieve_assistant(self, assistant_id: str) -> Assistant:
        """Retrieve a specific assistant."""
        response = self._get(f"assistants/{assistant_id}")
        return Assistant(**response)

    def delete_assistant(self, assistant_id: str) -> None:
        """Delete an assistant."""
        self._delete(f"assistants/{assistant_id}")

    def modify_assistant(
        self,
        assistant_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[AssistantTool]] = None,
        model: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Assistant:
        """Modify an existing assistant."""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if instructions is not None:
            data["instructions"] = instructions
        if tools is not None:
            data["tools"] = [tool.dict() for tool in tools]
        if model is not None:
            data["model"] = model
        if file_ids is not None:
            data["file_ids"] = file_ids
        if metadata is not None:
            data["metadata"] = metadata

        response = self._post(f"assistants/{assistant_id}", json=data)
        return Assistant(**response)

    def list_assistant_files(self, assistant_id: str) -> List[AssistantFile]:
        """List files associated with an assistant."""
        response = self._get(f"assistants/{assistant_id}/files")
        return [AssistantFile(**file) for file in response["data"]]

    def create_assistant_file(
        self,
        assistant_id: str,
        file_id: str
    ) -> AssistantFile:
        """Attach a file to an assistant."""
        response = self._post(
            f"assistants/{assistant_id}/files",
            json={"file_id": file_id}
        )
        return AssistantFile(**response)

    def delete_assistant_file(
        self,
        assistant_id: str,
        file_id: str
    ) -> None:
        """Remove a file from an assistant."""
        self._delete(f"assistants/{assistant_id}/files/{file_id}")

    # Threads API
    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> Thread:
        """Create a new thread."""
        data = {"metadata": metadata} if metadata else {}
        response = self._post("threads", json=data)
        return Thread(**response)

    def list_threads(
        self,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> ThreadListResponse:
        """List all threads with pagination."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get("threads", params=params)
        return ThreadListResponse(**response)

    def retrieve_thread(self, thread_id: str) -> Thread:
        """Retrieve a specific thread."""
        response = self._get(f"threads/{thread_id}")
        return Thread(**response)

    def modify_thread(
        self,
        thread_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Thread:
        """Modify an existing thread."""
        data = {"metadata": metadata} if metadata else {}
        response = self._post(f"threads/{thread_id}", json=data)
        return Thread(**response)

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread."""
        self._delete(f"threads/{thread_id}")

    # Messages API
    def create_message(
        self,
        thread_id: str,
        role: str,
        content: List[Dict[str, Any]],
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThreadMessage:
        """Create a message in a thread."""
        data = {
            "role": role,
            "content": content,
            "file_ids": file_ids or [],
            "metadata": metadata
        }
        response = self._post(f"threads/{thread_id}/messages", json=data)
        return ThreadMessage(**response)

    def list_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> MessageListResponse:
        """List messages in a thread with pagination."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get(f"threads/{thread_id}/messages", params=params)
        return MessageListResponse(**response)

    def retrieve_message(self, thread_id: str, message_id: str) -> ThreadMessage:
        """Retrieve a specific message."""
        response = self._get(f"threads/{thread_id}/messages/{message_id}")
        return ThreadMessage(**response)

    def modify_message(
        self,
        thread_id: str,
        message_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThreadMessage:
        """Modify an existing message."""
        data = {"metadata": metadata} if metadata else {}
        response = self._post(
            f"threads/{thread_id}/messages/{message_id}",
            json=data
        )
        return ThreadMessage(**response)

    # Runs API
    def create_run(
        self,
        thread_id: str,
        assistant_id: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Run:
        """Create a run in a thread."""
        data = {
            "assistant_id": assistant_id,
            "model": model,
            "instructions": instructions,
            "tools": tools,
            "metadata": metadata
        }
        response = self._post(f"threads/{thread_id}/runs", json=data)
        return Run(**response)

    def list_runs(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> RunListResponse:
        """List runs in a thread with pagination."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get(f"threads/{thread_id}/runs", params=params)
        return RunListResponse(**response)

    def retrieve_run(self, thread_id: str, run_id: str) -> Run:
        """Retrieve a specific run."""
        response = self._get(f"threads/{thread_id}/runs/{run_id}")
        return Run(**response)

    def modify_run(
        self,
        thread_id: str,
        run_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Run:
        """Modify an existing run."""
        data = {"metadata": metadata} if metadata else {}
        response = self._post(
            f"threads/{thread_id}/runs/{run_id}",
            json=data
        )
        return Run(**response)

    def cancel_run(self, thread_id: str, run_id: str) -> Run:
        """Cancel a run."""
        response = self._post(f"threads/{thread_id}/runs/{run_id}/cancel")
        return Run(**response)

    def submit_tool_outputs(
        self,
        thread_id: str,
        run_id: str,
        tool_outputs: List[Dict[str, Any]]
    ) -> Run:
        """Submit tool outputs for a run."""
        response = self._post(
            f"threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
            json={"tool_outputs": tool_outputs}
        )
        return Run(**response)

    # Files API
    def upload_file(
        self,
        file_path: str,
        purpose: str
    ) -> File:
        """Upload a file."""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"purpose": purpose}
            response = self._post("files", files=files, json=data)
        return File(**response)

    def list_files(
        self,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> FileListResponse:
        """List all files with pagination."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get("files", params=params)
        return FileListResponse(**response)

    def retrieve_file(self, file_id: str) -> File:
        """Retrieve a specific file."""
        response = self._get(f"files/{file_id}")
        return File(**response)

    def retrieve_file_content(self, file_id: str) -> bytes:
        """Retrieve the content of a file."""
        response = self._get(f"files/{file_id}/content")
        return response.content

    def delete_file(self, file_id: str) -> None:
        """Delete a file."""
        self._delete(f"files/{file_id}")

    # Fine-tuning API
    def create_fine_tuning_job(
        self,
        training_file: str,
        model: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> FineTuningJob:
        """Create a fine-tuning job."""
        data = {
            "training_file": training_file,
            "model": model,
            "validation_file": validation_file,
            "hyperparameters": hyperparameters or {}
        }
        response = self._post("fine_tuning/jobs", json=data)
        return FineTuningJob(**response)

    def list_fine_tuning_jobs(
        self,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> FineTuningJobListResponse:
        """List all fine-tuning jobs with pagination."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get("fine_tuning/jobs", params=params)
        return FineTuningJobListResponse(**response)

    def retrieve_fine_tuning_job(self, job_id: str) -> FineTuningJob:
        """Retrieve a specific fine-tuning job."""
        response = self._get(f"fine_tuning/jobs/{job_id}")
        return FineTuningJob(**response)

    def modify_fine_tuning_job(
        self,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FineTuningJob:
        """Modify an existing fine-tuning job."""
        data = {"metadata": metadata} if metadata else {}
        response = self._post(f"fine_tuning/jobs/{job_id}", json=data)
        return FineTuningJob(**response)

    def list_fine_tuning_events(
        self,
        job_id: str,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List events for a fine-tuning job."""
        params = {}
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before

        response = self._get(f"fine_tuning/jobs/{job_id}/events", params=params)
        return response["data"]

    # Moderation API
    def create_moderation(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None
    ) -> ModerationResponse:
        """Create a moderation check."""
        data = {"input": input}
        if model is not None:
            data["model"] = model

        response = self._post("moderations", json=data)
        return ModerationResponse(**response)

    # Image Generation API
    def create_image(
        self,
        prompt: str,
        n: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        response_format: Optional[str] = "url",
        user: Optional[str] = None
    ) -> ImageGenerationResponse:
        """Generate an image from a prompt."""
        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format
        }
        if user is not None:
            data["user"] = user

        response = self._post("images/generations", json=data)
        return ImageGenerationResponse(**response)

    def create_image_edit(
        self,
        image_path: str,
        prompt: str,
        mask_path: Optional[str] = None,
        n: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        response_format: Optional[str] = "url",
        user: Optional[str] = None
    ) -> ImageGenerationResponse:
        """Create an edited or extended image."""
        files = {"image": ("image.png", open(image_path, "rb"))}
        if mask_path:
            files["mask"] = ("mask.png", open(mask_path, "rb"))

        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format
        }
        if user is not None:
            data["user"] = user

        response = self._post("images/edits", files=files, json=data)
        return ImageGenerationResponse(**response)

    def create_image_variation(
        self,
        image_path: str,
        n: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        response_format: Optional[str] = "url",
        user: Optional[str] = None
    ) -> ImageGenerationResponse:
        """Create a variation of an image."""
        files = {"image": ("image.png", open(image_path, "rb"))}

        data = {
            "n": n,
            "size": size,
            "response_format": response_format
        }
        if user is not None:
            data["user"] = user

        response = self._post("images/variations", files=files, json=data)
        return ImageGenerationResponse(**response)

    # Audio API
    def create_transcription(
        self,
        file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = "json",
        temperature: Optional[float] = 0
    ) -> TranscriptionResponse:
        """Transcribe audio into text."""
        files = {"file": ("audio.mp3", open(file_path, "rb"))}

        data = {
            "model": model,
            "response_format": response_format,
            "temperature": temperature
        }
        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt

        response = self._post("audio/transcriptions", files=files, json=data)
        return TranscriptionResponse(**response)

    def create_translation(
        self,
        file_path: str,
        model: str = "whisper-1",
        prompt: Optional[str] = None,
        response_format: Optional[str] = "json",
        temperature: Optional[float] = 0
    ) -> TranslationResponse:
        """Translate audio into English."""
        files = {"file": ("audio.mp3", open(file_path, "rb"))}

        data = {
            "model": model,
            "response_format": response_format,
            "temperature": temperature
        }
        if prompt is not None:
            data["prompt"] = prompt

        response = self._post("audio/translations", files=files, json=data)
        return TranslationResponse(**response) 