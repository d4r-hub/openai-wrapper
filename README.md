# OpenAI Client Python

A Python client for interacting with the OpenAI API, including support for Assistants, Threads, Messages, Files, and Fine-tuning endpoints.

## Features

- Chat Completion
- Text Completion
- Embeddings
- Models
- Assistants API
- Threads API
- Messages API
- Files API
- Fine-tuning API

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd openai-client-python
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=http://localhost:8000/v1  # Optional, defaults to OpenAI's API
OPENAI_ORGANIZATION_ID=org-123  # Optional
```

## Usage

The client can be used as a Python module or run directly as a script.

### As a Module

```python
from openai_client.config.config import load
from openai_client.openai.api import OpenAIAPI
from openai_client.openai.models import Message

# Load configuration
cfg = load()
client = OpenAIAPI(cfg)

# Example: Chat completion
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Tell me a joke.")
]
response = client.chat_completion(messages)
print(response.choices[0].message.content)
```

### As a Script

Run the example script that demonstrates all API endpoints:

```bash
python -m openai_client
```

## Development

The project structure is organized as follows:

```
openai-client-python/
├── src/
│   └── openai_client/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config/
│       │   └── config.py
│       └── openai/
│           ├── api.py
│           ├── client.py
│           └── models.py
├── tests/
├── requirements.txt
└── README.md
```

## Testing

To run tests (when implemented):

```bash
python -m pytest tests/
```

## License

MIT License 