import openai
import os
import math
import json
import asyncio
import getpass
import time
import colorama
import sys
import re

try:
    import pyperclip
    has_imported_pyperclip = True
except ModuleNotFoundError:
    has_imported_pyperclip = False

colorama.init()

client = None
api_key = None

def set_api_key(key):
    global client
    client = openai.AsyncOpenAI(api_key=key)
    global api_key
    api_key = key

def _trim_at_higher_length(text, max_length, color=colorama.Fore.LIGHTBLACK_EX):
    stop_at_index = text.find('\n') 
    if stop_at_index == -1 or stop_at_index > max_length:
        stop_at_index = max_length
    
    if len(text) > stop_at_index:
        text = text[:stop_at_index] + color + '...' + colorama.Style.RESET_ALL
    return text

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
        content = _trim_at_higher_length(content, 100)
        
        spaces = (12 - len(role)) / 2
        print('\t' + brackets[0] + (math.ceil(spaces) * " ") + role + (math.floor(spaces) * " ") + brackets[1] + " " + content)

    async def get(self, user=None, system=None, messages=None, temperature=None, top_p=None, max_tokens=None, frequency_penalty=None, presence_penalty=None):
        global client
        global api_key
        try:
            if api_key == None:
                client = openai.AsyncOpenAI()
            else:
                client = openai.AsyncOpenAI(api_key=api_key)
        except openai.OpenAIError:
            print('No API key found in \"OPENAI_API_KEY\" environment variable.')
            print('You can input your API key here instead, though this is not recommended:')
            client = openai.AsyncOpenAI(api_key=(getpass.getpass('\tOpenAI API Key:')))

        if messages is None:
            messages = []

        if system is None:
            system = self.system

        use_system_property = False

        if system is not None:
            messages.insert(0, {'role': 'system', 'content': system})
            use_system_property = True
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
        if use_system_property:
            self.previous = self.previous[1:]

        message = response.choices[0].message
        self.previous.append({'role':message.role,'content':message.content})

        return response.choices[0].message.content

staticGpt = gpt()

async def get(user=None, use_previous=False, system=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    return await staticGpt.get(user=user, messages=(staticGpt.previous if use_previous else None), system=system, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

def reset(model='gpt-3.5-turbo'):
    staticGpt.previous = []
    staticGpt.model = model

def print_messages(conv, shorten_messages, additional_messages=None):
    if conv.system != None:
        print('{SYS} ' + conv.system)

    messages = conv.previous.copy()
    if additional_messages != None:
        messages += additional_messages

    for i in range(len(messages)):
        _print_message(message=messages[i], shorten_message=shorten_messages, i=i)

def _print_error(msg):
    print(colorama.Fore.RED + 'Error:\n\t' + msg + colorama.Style.RESET_ALL)

def _print_info(msg):
    print(colorama.Fore.GREEN + '( ' + msg + ' )' + colorama.Style.RESET_ALL)

def _print_message(message, shorten_message, i):
    brackets = '[]' if message['role'] == 'user' else \
        ('<>' if message['role'] == 'assistant' else '{}')

    lighter_color = colorama.Fore.LIGHTWHITE_EX if message['role'] == 'assistant' else colorama.Fore.LIGHTCYAN_EX
    darker_color = colorama.Fore.WHITE if message['role'] == 'assistant' else colorama.Fore.CYAN

    prefix = darker_color + brackets[0] + str(i) + brackets[1] + lighter_color + ' '

    if shorten_message:
        content = _trim_at_higher_length(message['content'], 100, color=darker_color)
        print(prefix + content + colorama.Style.RESET_ALL)
        return

    lines = message['content'].split('\n')

    print(prefix + lines[0])
    for i in range(1, len(lines)):
        print((' ' * (len(str(i)) + 3)) + lines[i])

    print(colorama.Style.RESET_ALL, end='')
    

async def _wait_for_response():
    start_time = time.time()
    while True:
        _print_info(f"Generating... [{math.floor((time.time() - start_time) * 10) / 10}s]")
        await asyncio.sleep(0.1)
        sys.stdout.write("\033[F")  # Cursor up one line
        sys.stdout.write("\033[K")  # Clear to the end of line


async def _get_response(conv, prompt):
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(conv.get(user=prompt, messages=conv.previous)), loop.create_task(_wait_for_response())]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        return task.result()

def _get_boolean_input(message:str, default_value:bool):
    while True:
        arg = input(message)
        if len(arg) == 0:
            return default_value

        arg = arg[0].lower()

        if arg == 'n':
            return False
        if arg == 'y':
            return True

saved_conversations = {}

def conversation(model='gpt-3.5-turbo', system=None, messages=None, user=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    conv = gpt(model=model, system=system, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    conversation_name = None

    if messages != None:
        conv.previous = messages

    if len(conv.previous) > 0 and conv.previous[0]['role'] == 'system':
        conv.system = conv.previous[0]['content']
        conv.previous.pop(0)

    full_view = True
    
    def reprint_conversation(additional_message=None):
        os.system('cls' if (os.name == 'nt') else 'clear')

        print(colorama.Fore.GREEN + 'Conversation started. Type ? for a list of commands.\n' + colorama.Style.RESET_ALL)
        print_messages(conv=conv, shorten_messages=(not full_view), additional_messages=None if additional_message == None else [additional_message])

    reprint_conversation()
        
    
    while True:
        if user != None:
            prompt = user
            user = None
            _print_message({'role':'user','content':prompt}, len(conv.previous))
        else:
            prompt = input(colorama.Fore.CYAN + f'[{len(conv.previous)}] ' + colorama.Fore.LIGHTCYAN_EX)

            if prompt == '?':
                print(colorama.Fore.GREEN + 'Commands:')
                print('\t[?] View list of commands')
                print('\t[!] Exit conversation')
                print('\t[:] Run Python command')
                print('\t[#] Set GPT\'s property (no argument to list properties, [?#] for list of properties, [##] to reset)')
                print('\t[$] Set system message')
                print('\t[+] Insert message before index (double + for assistant)')
                print('\t[-] Remove message at index (double - for clear conversation)')
                print('\t[~] Change message at index (double ~ for reverse role)')
                print('\t[&] Re-generate last GPT message')
                print('\t[@] Copy message to clipboard')
                print('\t[@~] Copy code block to clipboard')
                print('\t[@@] Copy conversation JSON to clipboard')
                print('\t[^] Save conversation')
                print('\t[%] Load conversation')
                print('\t[=] Switch between full and shortened view')
                print('\t[] Reload conversation')
                print('\t[\\] Override command')
                print('\t[_] Multiline (Ctrl+X with Enter to exit, Ctrl+C to cancel)' + colorama.Style.RESET_ALL)
                continue
            elif prompt == '?#':
                print(colorama.Fore.GREEN + '(#) Command help:')
                print('\t[model] / [m] The model to generate the completion')
                print('\t[temperature] / [temp] Controls randomness (0-2)')
                print('\t[max_tokens] / [max] The maximum number of tokens to generate')
                print('\t[top_p] / [top] Controls diversity via nucleus sampling (0-1)')
                print('\t[frequency_penalty] / [fp] Decreases token repetition (0-2)')
                print('\t[presence_penalty] / [pp] Increases likelihood of new topics (0-2)' + colorama.Style.RESET_ALL)
                continue
            elif prompt == '#':
                values = (('model', conv.model), ('temperature', conv.temperature), ('max_tokens', conv.max_tokens), ('top_p', conv.top_p), ('frequency_penalty', conv.frequency_penalty), ('presence_penalty', conv.presence_penalty))
                print(colorama.Fore.GREEN + '(#) Properties:')
                for element in values:
                    print(f'\t{element[0]}: {element[1]}')
                print(colorama.Style.RESET_ALL, end='')
                continue
            elif prompt == '##':
                conv.model = 'gpt-3.5-turbo'
                conv.temperature = 0
                conv.max_tokens = 2048
                conv.top_p = 0
                conv.frequency_penalty = 0
                conv.presence_penalty = 0
                _print_info('Reset all properties')
                continue
            
            if prompt == '':
                reprint_conversation()
                continue

            if prompt[0] != '\\':
                if prompt == '!':
                    return

                if prompt == '=':
                    full_view = not full_view
                    reprint_conversation()
                    continue

                elif prompt[0] == ':':
                    arg = prompt[1:]
                    exec(arg)
                    _print_info(f'Executed `{arg}`')
                    continue

                elif prompt[0] == '#':
                    space = prompt.find(' ')

                    if space == -1:
                        _print_error('No argument.')
                        continue

                    prop = prompt[1:space]
                    value = prompt[space+1:]

                    try:
                        match prop:
                            case 'model' | 'm':
                                conv.model = value
                            case 'temperature' | 'temp':
                                conv.temperature = float(value)
                            case 'max_tokens' | 'max':
                                conv.max_tokens = int(value)
                            case 'top_p' | 'top':
                                conv.top_p = float(value)
                            case 'frequency_penalty' | 'fp':
                                conv.frequency_penalty = float(value)
                            case 'presence_penalty' | 'pp':
                                conv.presence_penalty = float(value)
                            case _:
                                _print_error(f'`{prop}` is not a valid property.')
                                continue
                    except ValueError:
                        _print_error(f'`{value}` is not a valid value.')
                        continue
                

                    _print_info('Successfully set property')

                    continue

                elif prompt[0] == '$':
                    message = prompt[1:] if len(prompt) > 1 else None

                    conv.system = message

                    reprint_conversation()
                    continue

                elif prompt[0] == '+':
                    space = prompt.find(' ')
                    is_assistant = len(prompt) > 1 and prompt[1] == '+'
                    arg = prompt[space+1:]
                    valueString = prompt[(2 if is_assistant else 1):space]

                    try:
                        value = int(valueString)
                        conv.previous.insert(value, {'role': ('assistant' if is_assistant else 'user'), 'content': arg})
                        reprint_conversation()
                    except (IndexError, ValueError):
                        _print_error(f'`{valueString}` is not valid.')
                    continue
                
                elif prompt[0] == '-':
                    clear_everything = len(prompt) == 2 and prompt[1] == '-'

                    if clear_everything:
                        conv.previous = []
                        reprint_conversation()
                        continue

                    try:
                        value = int(prompt[1:])

                        conv.previous.pop(value)
                        reprint_conversation()
                    except (IndexError, ValueError):
                        _print_error(f'`{prompt[1:]}` is not a valid index.')
                    continue
                
                elif prompt[0] == '~':
                    space = prompt.find(' ')
                    change_role = len(prompt) > 1 and prompt[1] == '~'
                    try:
                        value = int(prompt[2:] if space == -1 else \
                                    prompt[(2 if change_role else 1):space])
                    except ValueError:
                        _print_error(f'Invalid value.')
                        continue


                    try:
                        new_role = conv.previous[value]['role']
                    except IndexError:
                        _print_error(f'No message with index {value}.')
                        continue
                    if change_role:
                        new_role = 'assistant' if new_role == 'user' else 'user'
                    
                    if space == -1 or len(prompt) <= space or prompt[space+1:].isspace():
                        if change_role:
                            conv.previous[value] = {'role': new_role, 'content':conv.previous[value]['content']}
                            reprint_conversation()
                            continue
                        _print_error('Body cannot be empty when single ~ used')
                        continue

                    arg = prompt[space+1:]


                    conv.previous[value] = { 'role': new_role, 'content': arg }
                    reprint_conversation()
                    continue

                elif prompt == '_': 
                    _print_info('Started multi-line. ^X to exit.')

                    prompt = ''
                    line_number = 1

                    try:
                        while True:
                            new_line = input(colorama.Style.RESET_ALL + (8 - len(str(line_number))) * ' ' + str(line_number) + '> ' + colorama.Fore.LIGHTBLUE_EX)
                            if new_line == '\x18':
                                break
                            if new_line == '\x15':
                                previous_line = prompt[:-1].rfind('\n')
                                if previous_line == -1:
                                    if prompt != '':
                                        previous_line = prompt[:-1].find('\n')
                                    else:
                                        _print_error('No previous line')
                                        continue

                                prompt = prompt[:(previous_line + 1)]
                                line_number -= 1

                                _print_info('Removed previous line')

                                continue
                            prompt += new_line + '\n'
                            line_number += 1
                    except KeyboardInterrupt:
                        print()
                        _print_info('Cancelled multi-line')
                        continue

                    prompt = prompt[:-1]

                    _print_info('Ended multi-line')

                    send_message = True
                    if prompt == '':
                        send_message = _get_boolean_input('Body is empty. Send anyways? (Y/n): ', True)

                    if not send_message:
                        continue
                
                elif prompt == '&':
                    if len(conv.previous) < 1:
                        _print_error('No messages.')
                        continue
                    index = -2 if conv.previous[-1]['role'] == 'assistant' else -1

                    prompt = conv.previous[index]['content']
                    conv.previous = conv.previous[:index]
                    reprint_conversation(additional_message={'role':'user','content':prompt})

                elif prompt[0] == '@':
                    global has_imported_pyperclip

                    if not has_imported_pyperclip:
                        _print_error('The pyperclip module is required to use clipboard\n\tInstall it using `pip install pyperclip`')
                        continue

                    if prompt == '@@':
                        content = json.dumps(conv.previous)
                        pyperclip.copy(content)
                        _print_info('Copied JSON to clipboard')
                        continue
                    
                    if len(prompt) >= 2 and prompt[:2] == '@~':
                        index = 0
                        try:
                            content = conv.previous[-1]['content']

                            if len(prompt) > 2:
                                index = int(prompt[2:]) - 1
                            
                            pattern = r"```.*?\n(.*?)```"
                            code_blocks = re.findall(pattern, content, re.DOTALL)

                            code_block = code_blocks[index]

                            pyperclip.copy(code_block)
                            _print_info(f'Copied codeblock {index + 1}')
                        except IndexError:
                            _print_error(f'No codeblock {index + 1}')
                        continue

                    if prompt[0] == '@':
                        if len(conv.previous) == 0:
                            _print_error('No lines')
                            continue

                        index = -1
                        try:
                            if len(prompt) > 1:
                                index = int(prompt[1:]) 
                            content = conv.previous[index]['content']
                        except (IndexError, ValueError):
                            print(f'Error: `{prompt[1:]}` is not a valid index')
                            continue

                        pyperclip.copy(content)
                        if index == -1:
                            _print_info('Copied last message to clipboard')
                        else:
                            _print_info(f'Copied message {index} to clipboard')
                        continue

                    _print_error('Invalid @ command')
                    continue

                elif prompt[0] == '%':
                    arg = prompt[1:]

                    if arg == '':
                        if len(saved_conversations) == 0:
                            print(colorama.Fore.GREEN + 'No saved conversations' + colorama.Style.RESET_ALL)
                            continue

                        print(colorama.Fore.GREEN + 'Saved conversations:')
                        for key in list(saved_conversations):
                            print('\t' + key)
                        print(colorama.Style.RESET_ALL, end='')
                        continue

                    value = saved_conversations.get(arg)

                    if value == None:
                        _print_error(f'Conversation "{arg}" not found. Type `%` for a list of saved conversations')
                        continue

                    conv.previous = value['messages']
                    model = value['model']
                    system = value['system']
                    temperature = value['temperature']
                    top_p = value['top_p']
                    max_tokens = value['max_tokens']
                    frequency_penalty = value['frequency_penalty']
                    presence_penalty = value['presence_penalty']

                    conversation_name = arg

                    reprint_conversation()
                    continue

                elif prompt[0] == '^':
                    arg = prompt[1:]
                    if conversation_name == None:
                        if arg == '':
                            _print_error('Current conversation is not currently saved under a name')
                            continue

                    if arg == '':
                        arg = conversation_name
                    
                    saved_conversations.setdefault(arg, {'messages': conv.previous, 'model': model, 'system':system, 'temperature':temperature, 'top_p': top_p, 'max_tokens': max_tokens, 'frequency_penalty':frequency_penalty, 'presence_penalty':presence_penalty})

                    conversation_name = arg

                    _print_info(f'Saved conversation as "{arg}"')
                    continue

            else:
                prompt = prompt[1:]

        cancel_sending = False
        while True:
            try:
                response = asyncio.run(_get_response(conv, prompt))
                break
            except KeyboardInterrupt:
                conv.previous = conv.previous[:-1]
                arg = ''

                if _get_boolean_input('Resend message? (Y/n): ', True):
                    continue

                cancel_sending = True
                break
            except openai.OpenAIError as e:
                print('OpenAI Error:\n\t' + e.message)
                conv.previous = conv.previous[:-1]
                cancel_sending = True
                break
    
        if not cancel_sending:
            _print_message({'role':'assistant','content':response}, (not full_view), len(conv.previous) - 1)

def c(model='gpt-3.5-turbo', system=None, messages=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    conversation(model=model, system=system, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

if __name__ == '__main__':  
    c()