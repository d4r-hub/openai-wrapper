from openai_client import OpenAIAPI, Message
import time

def main():
    print("Starting OpenAI Client test...")
    
    # Create the client with the actual API endpoint and key
    client = OpenAIAPI(
        api_key="",
        base_url="https://api.openai.com/v1"
    )
    print(f"Using endpoint: {client.base_url}")
    
    # Test chat completion
    print("\nTesting chat completion...")
    messages = [
        Message(role="user", content="write a haiku about ai")
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            model="gpt-4o-mini",
            store=True
        )
        print("\nChat response:")
        print(response.choices[0].message.content)
        print("\nUsage details:")
        print(f"- Total tokens: {response.usage.total_tokens}")
        print(f"- Prompt tokens: {response.usage.prompt_tokens}")
        print(f"- Completion tokens: {response.usage.completion_tokens}")
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")

if __name__ == "__main__":
    main() 