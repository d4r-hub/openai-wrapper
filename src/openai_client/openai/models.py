from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class TokenDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0
    reasoning_tokens: int = 0
    predicted_prediction_tokens: int = 0

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[TokenDetails] = None
    completion_tokens_details: Optional[TokenDetails] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]

class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ListModelsResponse(BaseModel):
    data: List[Model]
    object: str

class Assistant(BaseModel):
    id: str
    object: str
    created_at: int
    name: str
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None

class Thread(BaseModel):
    id: str
    object: str
    created_at: int
    metadata: Optional[Dict[str, Any]] = None

class ThreadMessage(BaseModel):
    id: str
    object: str
    created_at: int
    thread_id: str
    role: str
    content: List[Dict[str, Any]]
    file_ids: List[str] = []
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Run(BaseModel):
    id: str
    object: str
    created_at: int
    thread_id: str
    assistant_id: str
    status: str
    required_action: Optional[Dict[str, Any]] = None
    last_error: Optional[Dict[str, Any]] = None
    expires_at: Optional[int] = None
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class File(BaseModel):
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = None
    status_details: Optional[Dict[str, Any]] = None

class FineTuningJob(BaseModel):
    id: str
    object: str
    created_at: int
    error: Optional[Dict[str, Any]] = None
    fine_tuned_model: Optional[str] = None
    finished_at: Optional[int] = None
    hyperparameters: Dict[str, Any]
    model: str
    organization_id: str
    result_files: List[str]
    status: str
    trained_tokens: Optional[int] = None
    training_file: str
    validation_file: Optional[str] = None

class Tool(BaseModel):
    type: str
    function: Dict[str, Any]

class AssistantTool(BaseModel):
    type: str
    function: Optional[Dict[str, Any]] = None
    code_interpreter: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None

class AssistantFile(BaseModel):
    id: str
    object: str
    created_at: int
    assistant_id: str

class AssistantListResponse(BaseModel):
    object: str
    data: List[Assistant]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class ThreadListResponse(BaseModel):
    object: str
    data: List[Thread]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class MessageListResponse(BaseModel):
    object: str
    data: List[ThreadMessage]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class RunListResponse(BaseModel):
    object: str
    data: List[Run]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class FileListResponse(BaseModel):
    object: str
    data: List[File]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class FineTuningJobListResponse(BaseModel):
    object: str
    data: List[FineTuningJob]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class CreateAssistantRequest(BaseModel):
    name: str
    model: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[AssistantTool]] = None
    file_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class CreateThreadRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

class CreateMessageRequest(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    file_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class CreateRunRequest(BaseModel):
    assistant_id: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[AssistantTool]] = None
    metadata: Optional[Dict[str, Any]] = None

class CreateFineTuningJobRequest(BaseModel):
    training_file: str
    model: str
    validation_file: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    suffix: Optional[str] = None

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class ModerationCategoryScores(BaseModel):
    hate: float
    hate_threatening: float
    self_harm: float
    sexual: float
    sexual_minors: float
    violence: float
    violence_graphic: float

class ModerationCategories(BaseModel):
    hate: bool
    hate_threatening: bool
    self_harm: bool
    sexual: bool
    sexual_minors: bool
    violence: bool
    violence_graphic: bool

class ModerationResult(BaseModel):
    categories: ModerationCategories
    category_scores: ModerationCategoryScores
    flagged: bool

class ModerationResponse(BaseModel):
    id: str
    model: str
    results: List[ModerationResult]

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int
    data: List[Dict[str, Any]]

class ImageEditRequest(BaseModel):
    image: str
    mask: Optional[str] = None
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None

class ImageVariationRequest(BaseModel):
    image: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    user: Optional[str] = None

class TranscriptionRequest(BaseModel):
    file: str
    model: str = "whisper-1"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0

class TranscriptionResponse(BaseModel):
    text: str

class TranslationRequest(BaseModel):
    file: str
    model: str = "whisper-1"
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0

class TranslationResponse(BaseModel):
    text: str 