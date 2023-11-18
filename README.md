# easygpt

`easygpt` is a Python library designed to simplify the interaction with OpenAI's GPT (Generative Pre-trained Transformer) models. It provides a convenient interface for sending prompts to the model and receiving responses, with optional logging for debugging purposes.

## Installation

You can install `easygpt` via pip:
```bash
pip install easygpt
```

## Usage

To use `easygpt`, you need to have an API key from OpenAI. This key must be set as an environment variable `OpenAI_APIKey` before using the library.

### Initialization

First, import the `easygpt` module and initialize the `gpt` class:

```python
import easygpt

gpt = easygpt.gpt()
```

The `gpt` class constructor accepts the following parameters:

- `model`: The identifier of the GPT model to use (default: `'gpt-3.5-turbo'`).
- `system`: An optional system-level prompt that provides instructions or context for the GPT model.
- `temperature`: Controls randomness in the response generation (default: `0`).
- `top_p`: Controls diversity of the response generation (default: `0`).
- `max_tokens`: The maximum number of tokens to generate in the response (default: `2048`).
- `frequency_penalty`: Decreases the likelihood of repetition in the response (default: `0`).
- `presence_penalty`: Encourages the model to talk about new topics (default: `0`).
- `logs`: Enables or disables logging of the interaction (default: `False`).

### Sending Prompts

To send a prompt to the GPT model, use the `get` method of the `gpt` instance:

```python
response = gpt.get(user="Your prompt here")
```

The `get` method accepts the following parameters:

- `system`: Overrides the system-level prompt for this request.
- `user`: The user-level prompt to send to the model.
- `messages`: A list of previous message exchanges to maintain context.
- `temperature`: Overrides the default temperature for this request.
- `top_p`: Overrides the default top_p for this request.
- `max_tokens`: Overrides the default max_tokens for this request.
- `frequency_penalty`: Overrides the default frequency_penalty for this request.
- `presence_penalty`: Overrides the default presence_penalty for this request.

Since your last request is stored under `self.previous`, you can append a message to your conversation like that:

```python
response = gpt.get(messages=gpt.previous, user="Another message")
```

### Logging

If logging is enabled, `easygpt` will print the interaction with the GPT model to the console. This includes the prompts sent by the user and system, as well as the responses from the assistant.

## Notes

- The `easygpt` library assumes that the OpenAI API key is set in the environment variable `OpenAI_APIKey`.
- The `gpt` class maintains a history of the conversation in the `previous` attribute, which can be used to provide context for subsequent requests.
