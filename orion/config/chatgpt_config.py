class ChatGPTConfig:
    pass


####################### OpenAI #######################
class OpenAIConfig(ChatGPTConfig):
    api_type: str = "openai"
    api_key: str = "<API_KEY>"
    model: str
    limit: int
    price: float


class OpenAIGPT35Config(OpenAIConfig):
    model: str = "gpt-3.5-turbo-0125"
    limit: int = 16000
    price: float = 0.0005


class OpenAIGPT4Config(OpenAIConfig):
    model: str = "gpt-4-turbo-preview"
    limit: int = 128000
    price: float = 0.01


####################### Azure #######################


class AzureConfig(ChatGPTConfig):
    api_type: str = "azure"
    api_key: str = "<API_KEY>"
    api_version: str = "2023-12-01-preview"
    azure_endpoint: str = "<ENDPOINT>"
    model: str
    limit: int
    price: float


class AzureGPT35Config(AzureConfig):
    model: str = "gpt-35-turbo-16k-0613"
    limit: int = 16000
    price: float = 0.0005


class AzureGPT4Config(AzureConfig):
    model: str = "gpt-4-0125-preview"
    limit: int = 128000
    price: float = 0.01
