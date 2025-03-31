import os
from .config.config import load
from .openai.api import OpenAIAPI
from .openai.models import Message

def main():
    # Load configuration
    cfg = load()
    client = OpenAIAPI(cfg)

    # Test chat completion
    print("\n=== Testing Chat Completion ===")
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Tell me a joke.")
    ]
    chat_resp = client.chat_completion(messages)
    print("Chat Response:", chat_resp.choices[0].message.content)

    # Test text completion
    print("\n=== Testing Text Completion ===")
    completion_resp = client.completion(
        prompt="Write a short story about a robot learning to paint.",
        model="text-davinci-003"
    )
    print("Completion Response:", completion_resp.choices[0].text)

    # Test embeddings
    print("\n=== Testing Embeddings ===")
    embedding_resp = client.create_embeddings(
        input=["Hello, world!", "This is a test."],
        model="text-embedding-ada-002"
    )
    print(f"Created {len(embedding_resp.data)} embeddings")

    # Test listing models
    print("\n=== Testing List Models ===")
    models_resp = client.list_models()
    print("Available models:")
    for model in models_resp.data:
        print(f"- {model.id}")

    # Test retrieving a specific model
    print("\n=== Testing Retrieve Model ===")
    model = client.retrieve_model("gpt-4-turbo")
    print(f"Retrieved model: {model.id} (owned by {model.owned_by})")

    # Test Assistants API
    print("\n=== Testing Assistants API ===")
    
    # Create an assistant
    assistant = client.create_assistant(
        name="Math Tutor",
        model="gpt-4-turbo",
        description="A helpful math tutoring assistant",
        instructions="You are a helpful math tutor. Explain concepts clearly and provide examples."
    )
    print(f"Created assistant: {assistant.id}")

    # List assistants
    assistants = client.list_assistants()
    print("Available assistants:")
    for a in assistants:
        print(f"- {a.name} ({a.id})")

    # Test Threads API
    print("\n=== Testing Threads API ===")
    
    # Create a thread
    thread = client.create_thread(metadata={"subject": "Math tutoring session"})
    print(f"Created thread: {thread.id}")

    # Create a message in the thread
    message = client.create_message(
        thread_id=thread.id,
        role="user",
        content=[{"type": "text", "text": "Can you help me understand quadratic equations?"}]
    )
    print(f"Created message: {message.id}")

    # List messages in the thread
    messages = client.list_messages(thread.id)
    print("Messages in thread:")
    for m in messages:
        print(f"- {m.role}: {m.content[0]['text']}")

    # Create a run
    run = client.create_run(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    print(f"Created run: {run.id} (status: {run.status})")

    # List runs
    runs = client.list_runs(thread.id)
    print("Runs in thread:")
    for r in runs:
        print(f"- {r.id}: {r.status}")

    # Test Files API
    print("\n=== Testing Files API ===")
    
    # Create a test file
    test_file = "test_data.jsonl"
    with open(test_file, "w") as f:
        f.write('{"prompt": "Hello", "completion": "Hi there!"}\n')
        f.write('{"prompt": "How are you?", "completion": "I\'m doing well!"}\n')

    try:
        # Upload the file
        file = client.upload_file(test_file, "fine-tune")
        print(f"Uploaded file: {file.filename} (ID: {file.id})")

        # List files
        files = client.list_files()
        print("Available files:")
        for f in files:
            print(f"- {f.filename} (ID: {f.id})")

        # Retrieve file details
        retrieved_file = client.retrieve_file(file.id)
        print(f"Retrieved file details: {retrieved_file.filename} (Purpose: {retrieved_file.purpose})")

        # Delete the file
        client.delete_file(file.id)
        print(f"Deleted file: {file.id}")

    finally:
        # Clean up test file
        os.remove(test_file)

    # Test Fine-tuning API
    print("\n=== Testing Fine-tuning API ===")
    
    # Create a fine-tuning job
    ft_job = client.create_fine_tuning_job(
        training_file="file_123",  # In a real scenario, this would be a valid file ID
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": "auto"}
    )
    print(f"Created fine-tuning job: {ft_job.id} (Status: {ft_job.status})")

    # List fine-tuning jobs
    ft_jobs = client.list_fine_tuning_jobs()
    print("Fine-tuning jobs:")
    for job in ft_jobs:
        print(f"- {job.id}: {job.status}")

    # Retrieve fine-tuning job details
    retrieved_job = client.retrieve_fine_tuning_job(ft_job.id)
    print(f"Retrieved job details: {retrieved_job.id} (Model: {retrieved_job.model})")

    # Cancel the fine-tuning job
    cancelled_job = client.cancel_fine_tuning_job(ft_job.id)
    print(f"Cancelled job: {cancelled_job.id} (Status: {cancelled_job.status})")

    # Clean up
    print("\n=== Cleaning Up ===")
    client.delete_assistant(assistant.id)
    print(f"Deleted assistant: {assistant.id}")

    client.delete_thread(thread.id)
    print(f"Deleted thread: {thread.id}")

if __name__ == "__main__":
    main() 