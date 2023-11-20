import openai
import os
import math
import json
import asyncio
import getpass

try:
    import pyperclip
    has_imported_pyperclip = True
except ModuleNotFoundError:
    has_imported_pyperclip = False

client = None

def set_api_key(api_key):
    global client
    client = openai.AsyncOpenAI(api_key=api_key)

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

    async def get(self, user=None, system=None, messages=None, temperature=None, top_p=None, max_tokens=None, frequency_penalty=None, presence_penalty=None):
        global client
        if client is None:
            try:
                client = openai.AsyncOpenAI()
            except openai.OpenAIError:
                print('No API key found in \"OPENAI_API_KEY\" environment variable.')
                print('You can input your API key here instead, though this is not recommended:')
                client = openai.AsyncOpenAI(api_key=(getpass.getpass('\tOpenAI API Key:')))

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
        
        response = await client.chat.completions.create(
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
        message = response.choices[0].message
        self.previous.append({'role':message.role,'content':message.content})

        return response.choices[0].message.content

staticGpt = gpt()

async def get(user=None, use_previous=False, system=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    return await staticGpt.get(user=user, messages=(staticGpt.previous if use_previous else None), system=system, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

def reset(model='gpt-3.5-turbo'):
    staticGpt.previous = []
    staticGpt.model = model

def print_messages(messages):
    for i in range(len(messages)):
        brackets = '[]' if messages[i]['role'] == 'user' else \
            ('<>' if messages[i]['role'] == 'assistant' else '{}')

        lines = messages[i]['content'].split('\n')

        prefix = brackets[0] + str(i) + brackets[1] + ' '

        print(prefix + lines[0])
        for i in range(1, len(lines)):
            print((' ' * len(prefix)) + lines[i])

async def conversation(model='gpt-3.5-turbo', system=None, messages=None, user=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    conv = gpt(model=model, system=system, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    if messages != None:
        conv.previous = messages
    
    print('Conversation started. Type ? for a list of commands.\n')

    if len(conv.previous) > 0 and conv.previous[0]['role'] == 'system':
        system = conv.previous[0]['content']
        conv.previous.pop(0)

    if system != None:
        print('{SYS} ' + system)
    
    def reprint_conversation(additional_message=None):
        os.system('cls' if (os.name == 'nt') else 'clear')
        print_messages(messages=((conv.previous + [additional_message]) if additional_message != None else conv.previous))

    reprint_conversation()
        
    
    while True:
        if user != None:
            prompt = user
            user = None
            print(f'[{len(conv.previous)}] {prompt}')
        else:
            prompt = input(f'[{len(conv.previous)}] ')

            if prompt == '?':
                print('Commands:')
                print('\t[?] View list of commands')
                print('\t[!] Exit conversation')
                print('\t[:] Run Python command')
                print('\t[#] Set GPT\'s property')
                print('\t[+] Insert message before index (double + for assistant)')
                print('\t[-] Remove message at index')
                print('\t[~] Change message at index (double ~ for reverse role)')
                print('\t[&] Re-generate last GPT message')
                print('\t[@] Copy last message to clipboard')
                print('\t[@@] Copy conversation JSON to clipboard')
                print('\t[] Reload conversation')
                print('\t[\\] Override command')
                print('\t[_] Multiline (Ctrl+X with Enter to exit)')
                continue

            
            if prompt == '':
                reprint_conversation()
                continue

            if prompt[0] != '\\':
                if prompt == '!':
                    return

                elif prompt[0] == ':':
                    arg = prompt[1:]
                    exec(arg)
                    print(f'( Executed `{arg}` )')
                    continue

                elif prompt[0] == '#':
                    space = prompt.find(' ')
                    arg = f'conv.{prompt[1:space]}={prompt[space+1:]}'

                    exec(arg)
                    print(f'( Executed `{arg}` )')
                    reprint_conversation()
                    continue

                elif prompt[0] == '+':
                    space = prompt.find(' ')
                    is_assistant = len(prompt) > 1 and prompt[1] == '+'
                    value = int(prompt[(2 if is_assistant else 1):space])
                    arg = prompt[space+1:]

                    conv.previous.insert(value, {'role': ('assistant' if is_assistant else 'user'), 'content': arg})
                    reprint_conversation()
                    continue
                
                elif prompt[0] == '-':
                    clear_everything = len(prompt) == 2 and prompt[1] == '-'

                    if clear_everything:
                        conv.previous = []
                        reprint_conversation()
                        continue

                    value = int(prompt[1:])

                    conv.previous.pop(value)
                    reprint_conversation()
                    continue

                elif prompt[0] == '~':
                    space = prompt.find(' ')
                    change_role = len(prompt) > 1 and prompt[1] == '~'
                    value = int(prompt[2:] if space == -1 else \
                                 prompt[(2 if change_role else 1):space])

                    new_role = conv.previous[value]['role']
                    if change_role:
                        new_role = 'assistant' if new_role == 'user' else 'user'
                    
                    if space == -1 or len(prompt) <= space or prompt[space+1:].isspace():
                        if change_role:
                            conv.previous[value] = {'role': new_role, 'content':conv.previous[value]['content']}
                            reprint_conversation()
                            continue
                        print('Error:\n\tBody cannot be empty when single ~ used')
                        continue

                    arg = prompt[space+1:]


                    conv.previous[value] = { 'role': new_role, 'content': arg }
                    reprint_conversation()
                    continue

                elif prompt == '_': 
                    print('( Started multi-line. ^X to exit. )')

                    prompt = ''
                    line_number = 1

                    while True:
                        new_line = input((8 - len(str(line_number))) * ' ' + str(line_number) + '> ')
                        if new_line == '\x18':
                            break
                        if new_line == '\x15':
                            previous_line = prompt[:-1].rfind('\n')
                            if previous_line == -1:
                                print('Error:\n\tNo previous line')
                                continue

                            prompt = prompt[:(previous_line + 1)]
                            line_number -= 1

                            print('( Removed previous line )')

                            continue
                        prompt += new_line + '\n'
                        line_number += 1

                    prompt = prompt[:-1]
                    print('( Ended multi-line )')
                
                elif prompt == '&':
                    if len(conv.previous) < 2 or conv.previous[-1]['role'] != 'assistant':
                        print('Error:\n\tMust have at least two messages and last message must be an assistant message.')
                        continue
                    prompt = conv.previous[-2]['content']
                    conv.previous = conv.previous[:-2]
                    reprint_conversation(additional_message={'role':'user','content':prompt})

                elif prompt == '@' or prompt == '@@':
                    global has_imported_pyperclip

                    if not has_imported_pyperclip:
                        print('Error:\n\tThe pyperclip module is required to use clipboard\n\tInstall it using `pip install pyperclip`')
                        continue
                    
                    if prompt == '@':
                        content = conv.previous[-1]['content']
                        pyperclip.copy(content)
                        print('( Copied last message to clipboard )')
                        continue
                    if prompt == '@@':
                        content = json.dumps(conv.previous)
                        pyperclip.copy(content)
                        print('( Copied JSON to clipboard )')
                        continue
            else:
                prompt = prompt[1:]

        response = await conv.get(user=prompt, messages=conv.previous)
        print(f'<{len(conv.previous) - 1}> ' + response)

def convo(model='gpt-3.5-turbo', system=None, messages=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    asyncio.run(conversation(model=model, system=system, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty))

if __name__ == '__main__':  
    convo()