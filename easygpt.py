import openai
import os

openai.api_key = os.environ.get('OpenAI_APIKey')

class gpt:
    def __init__(self, model='gpt-3.5-turbo', temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.previous = []

    def get(self, system=None, user=None, messages=[], temperature=None, top_p=None, max_tokens=None, frequency_penalty=None, presence_penalty=None):
        if system is not None:
            messages.insert(0, {'role': 'system', 'content': system})
        if user is not None:
            messages.append({'role': 'user', 'content': user})

        # Use instance defaults if not overridden
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.frequency_penalty
        presence_penalty = presence_penalty if presence_penalty is not None else self.presence_penalty

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        self.previous = messages
        self.previous.append(response.choices[0].message)

        return response.choices[0].message.content
