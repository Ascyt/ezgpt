# ezgpt

`ezgpt` is a Python library designed to simplify the interaction with OpenAI's GPT (Generative Pre-trained Transformer) models. It provides a convenient interface for sending prompts to the model and receiving responses, with optional logging for debugging purposes.

## Installation

You can install `ezgpt` via pip:
```bash
pip install ezgpt
```

## Usage

To use `ezgpt`, you need to have an API key from OpenAI. This key must either be set as an environment variable `OPENAI_API_KEY` or be passed with the static function `ezgpt.set_api_key(api_key)` before using the library. Since these are async functions, you either have to use `await`, or `import asyncio` and use `asyncio.run()`.

### Initialization

First, import the `ezgpt` module and initialize the `gpt` class:

```python
import ezgpt

gpt = ezgpt.gpt()
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
response = await gpt.get(user="Your prompt here")
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
response = await gpt.get(messages=gpt.previous, user="Another message")
```

### Logging

If logging is enabled, `ezgpt` will print the interaction with the GPT model to the console. This includes the prompts sent by the user and system, as well as the responses from the assistant.

## Simpler usage
If you don't care about any classes or instances, you can also just use the static `get` and `reset` functions:

```python
await ezgpt.get('Hello!') 
# 'Hello! How can I assist you today?'
```
Aside the normal GPT arguments, there's another boolean `use_previous` to use the previous conversation:
```python
await ezgpt.get('What was my previous message?', True) 
# 'Your previous message was "Hello".'
```

You can use the `reset()` function to clear the previous messages, and optionally also change the model:
```python
ezgpt.reset('gpt-4')
```

## Conversation

An even simpler way to use GPT is using the `conversation()` static function. Here you can use the normal GPT arguments, and when you use it, you'll be able to have a conversation with GPT. 

There are special commands you can use:

- `?`: View the list of commands
- `!`: Exit the conversation
- `:[command]`: Run Python command: This will execute everything after the `:` as a line of Python. 
    -   *Example: `:print('hello')`* will print `hello`
- `#[property] [value]`: Set GPT's property, such as `model`. Use just `#` to list all properties. Use `##` to reset all properties. Use `?#` for help.
    - *Example: `#model gpt-4`* will set the model to `gpt-4`
- `$[message]`: Set the system message. Leave message empty to remove.
- `+[index] [message]`: Insert message before index (double `+` for assistant instead of user)
    - *Example: `++0 Hello!`* will insert a message at the start by the assistant with content `Hello!`
- `-[index]`: Remove message at index (double `-` for clear everything)
    - *Example: `-0` will remove the first message*
- `~[index] [message]`: Change message at index (double `~` for reverse role). If you use `~~` you don't need a message argument.
    - *Example: `~~0 Hello!` will change the first message to `Hello!` and switch its role*
- `&`: Re-generates the last message GPT sent. 
- `@<index>`: Copy message at index (defaults to last message) to clipboard (requires `pyperclip` module)
- `@~<index>`: Copy codeblock. Index starts and defaults to `1` (first codeblock).
- `@@`: Copy conversation JSON to clipboard (requires `pyperclip` module)
- `=`: Switch between full view and shortened view. Shortened view will only contain the first line up to 100 characters for easy readability. 
- *empty*: Sending an empty message will reload the conversation to only include the actual messages. 
- `\[message]`: This will let you type messages with special characters at the beginning without having them act as commands.
    - *Example: `\- Hello` will add message `- Hello`*
- `_`: Start multiline. This lets you write and paste in multi-line text. `Ctrl+X` with `Enter` to stop multiline, `Ctrl+C` to cancel, `Ctrl+U` with `Enter` to remove previous line.

A shorthand to start `conversation` is using the `ezgpt.c()` static function. While generating, you can use `Ctrl+C` to cancel or resend.

## Notes

- The `ezgpt` library assumes that the OpenAI API key is set in the environment variable `OPENAI_API_KEY` or has been passed using `ezgpt.set_api_key(api_key)`.
- The `gpt` class maintains a history of the conversation in the `previous` attribute, which can be used to provide context for subsequent requests.
