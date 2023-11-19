import openai
import os
import math

openai.api_key = os.environ.get('OpenAI_APIKey')

class gpt:
    def __init__(self, model='gpt-3.5-turbo', system=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0, logs=False):
        self.model = model
        self.system = system
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.previous = []
        self.logs = logs

    def _print_log(self, role, content, brackets):
        MAX_LENGTH = 100
        stop_at_index = content.find('\n') 
        if stop_at_index == -1 or stop_at_index > MAX_LENGTH:
            stop_at_index = MAX_LENGTH    
        
        if len(content) > stop_at_index:
            content = content[:stop_at_index] + '...'
        
        spaces = (12 - len(role)) / 2
        print('\t' + brackets[0] + (math.ceil(spaces) * " ") + role + (math.floor(spaces) * " ") + brackets[1] + " " + content)

    def get(self, user=None, system=None, messages=None, temperature=None, top_p=None, max_tokens=None, frequency_penalty=None, presence_penalty=None):
        if messages is None:
            messages = []

        if system is None:
            system = self.system

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

        if self.logs:
          for message in messages:
              self._print_log(message['role'], message['content'], '[]')
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        if self.logs:
            self._print_log('assistant', response.choices[0].message.content, '<>')

        self.previous = messages
        self.previous.append(response.choices[0].message)

        return response.choices[0].message.content

staticGpt = gpt()

def get(user=None, use_previous=False, system=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    return staticGpt.get(user=user, messages=(staticGpt.previous if use_previous else None), system=system, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

def reset(model='gpt-3.5-turbo'):
    staticGpt.previous = []
    staticGpt.model = model