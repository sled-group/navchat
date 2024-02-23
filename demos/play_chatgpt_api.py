import argparse

from orion.chatgpt.api import ChatAPI
from orion.config.chatgpt_config import (
    AzureGPT4Config,
    AzureGPT35Config,
    OpenAIGPT4Config,
    OpenAIGPT35Config,
)

parser = argparse.ArgumentParser()
parser.add_argument("--api-type", choices=["openai", "azure"], default="azure")
parser.add_argument("--model-type", choices=["gpt35", "gpt4"], default="gpt4")
parser.add_argument("--stream", action="store_true", default=False)

args = parser.parse_args()

if args.api_type == "openai":
    if args.model_type == "gpt35":
        chat_api = ChatAPI(config=OpenAIGPT35Config())
    elif args.model_type == "gpt4":
        chat_api = ChatAPI(config=OpenAIGPT4Config())
    else:
        raise ValueError("model_type can only be ['gpt35', 'gpt4']")
elif args.api_type == "azure":
    if args.model_type == "gpt35":
        chat_api = ChatAPI(config=AzureGPT35Config())
    elif args.model_type == "gpt4":
        chat_api = ChatAPI(config=AzureGPT4Config())
    else:
        raise ValueError("model_type can only be ['gpt35', 'gpt4']")

while True:
    utter = input("\nUser>>>")
    chat_api.add_user_message(utter)
    if args.stream:
        response = ""
        gen = chat_api.get_system_response_stream()
        print("Response>>>", end="")
        for chuck in gen:
            response += chuck
            print(chuck, end="")
        print()
    else:
        response = chat_api.get_system_response()
        print("Response>>>", response)
    chat_api.add_assistant_message(response)
