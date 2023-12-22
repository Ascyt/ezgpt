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

# Has to also be updated in ../setup.py because I'm too lazy to make that work
VERSION = '1.15.2'

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
        _print_message({'role':'system','content':conv.system}, shorten_message=shorten_messages, i=-1)

    messages = conv.previous.copy()
    if additional_messages != None:
        messages += additional_messages

    for i in range(len(messages)):
        _print_message(message=messages[i], shorten_message=shorten_messages, i=i)

def _print_error(msg):
    print(colorama.Fore.RED + 'Error:\n\t' + msg + colorama.Style.RESET_ALL)

def _print_info(msg):
    print(colorama.Fore.LIGHTGREEN_EX + '( ' + msg + ' )' + colorama.Style.RESET_ALL)

def _print_message(message, shorten_message, i):
    match (message['role']):
        case 'assistant': 
            light = colorama.Fore.WHITE
            dark = colorama.Fore.LIGHTBLACK_EX
            brackets = '<>'
        case 'user':
            light = colorama.Fore.LIGHTCYAN_EX 
            dark = colorama.Fore.CYAN
            brackets = '[]'
        case 'system':
            light = colorama.Fore.LIGHTYELLOW_EX
            dark = colorama.Fore.YELLOW
            brackets = '{}'
        case _:
            light = colorama.Fore.WHITE
            dark = colorama.Fore.LIGHTBLACK_EX
            brackets = '()'
    
    value = 'S' if (i == -1 and message['role'] == 'system') else str(i)
    prefix = dark + brackets[0] + value + brackets[1] + light + ' '

    if shorten_message:
        content = _trim_at_higher_length(message['content'], 100, color=dark)
        print(prefix + content + colorama.Style.RESET_ALL)
        return

    code_index_counter = 0
    def get_code_index_counter():
        nonlocal code_index_counter
        code_index_counter += 1
        return str(code_index_counter)

    pattern = r"```.*?\n(.*?)```"
    content = message['content']
    content = re.sub(pattern, lambda match: colorama.Fore.MAGENTA + get_code_index_counter() + ':' + dark + match.group() + light, content, flags=re.DOTALL)

    lines = content.split('\n')

    print(prefix + lines[0])
    for j in range(1, len(lines)):
        print((' ' * ((1 if i == -1 else len(str(i))) + 3)) + lines[j])

    print(colorama.Style.RESET_ALL, end='')
    

async def _wait_for_response():
    start_time = time.time()
    while True:
        _print_info(f"Generating... [{math.floor((time.time() - start_time) * 10) / 10}s]")
        await asyncio.sleep(0.1)
        sys.stdout.write("\033[F")  # Cursor up one line


async def _get_response(conv, prompt):
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(conv.get(user=prompt, messages=conv.previous)), loop.create_task(_wait_for_response())]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        return task.result()

def _get_boolean_input(message:str, default_value:bool):
    while True:
        arg = input(colorama.Fore.CYAN + message + colorama.Fore.LIGHTCYAN_EX)
        if len(arg) == 0:
            return default_value

        arg = arg[0].lower()

        if arg == 'n':
            return False
        if arg == 'y':
            return True

def _get_multiline(role):
    _print_info('Started multi-line. ^X to exit')

    match (role):
        case 'assistant': 
            light = colorama.Fore.WHITE
            dark = colorama.Fore.LIGHTBLACK_EX
        case 'user':
            light = colorama.Fore.LIGHTCYAN_EX 
            dark = colorama.Fore.CYAN
        case 'system':
            light = colorama.Fore.LIGHTYELLOW_EX
            dark = colorama.Fore.YELLOW
        case _:
            light = colorama.Fore.WHITE
            dark = colorama.Fore.LIGHTBLACK_EX

    prompt = ''
    line_number = 1

    try:
        while True:
            new_line = input(dark + (8 - len(str(line_number))) * ' ' + str(line_number) + '> ' + light)
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
        return None

    prompt = prompt[:-1]

    _print_info('Ended multi-line')

    send_message = True
    if prompt == '':
        send_message = _get_boolean_input('Body is empty. Send anyways? (Y/n): ', True)

    if not send_message:
        return None

    return prompt


saved_conversations = {}
PERSISTENT_CONVERSATION_PATH = os.path.join(os.path.expanduser('~'), '.ezgpt_conversations.json')
persistant_saved_conversations = {}

if os.path.exists(PERSISTENT_CONVERSATION_PATH):
    with open(PERSISTENT_CONVERSATION_PATH, 'r') as file:
        persistant_saved_conversations = json.load(file)

def save_conversation_persistant(name, conversation):
    persistant_saved_conversations.setdefault(name, conversation)

    update_persistant()

def update_persistant():
    with open(PERSISTENT_CONVERSATION_PATH, 'w') as file:
        json.dump(persistant_saved_conversations, file)

def conversation(model='gpt-3.5-turbo', system=None, messages=None, user=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    global has_imported_pyperclip

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

        print(colorama.Fore.GREEN + f'ezgpt.conversation v{VERSION} | Type ? for a list of commands\n' + colorama.Style.RESET_ALL)
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
                print(colorama.Fore.LIGHTGREEN_EX + 'Commands:')
                print('\t[?] View list of commands')
                print('\t[??] Print general information')
                print('\t')
                print('\t[!] Exit conversation')
                print('\t[:] Run Python command')
                print('\t[#] Set GPT\'s property (no argument to list properties, [?#] for list of properties, [##] to reset)')
                print('\t[$] Set system message')
                print('\t')
                print('\t[+] Insert message before index (double + for assistant)')
                print('\t[-] Remove message at index (double - for clear conversation)')
                print('\t[~] Change message at index (double ~ for reverse role)')
                print('\t')
                print('\t[@] Copy message to clipboard')
                print('\t[@~] Copy code block to clipboard')
                print('\t[@@] Copy conversation JSON to clipboard')
                print('\t')
                print('\t[^] Save conversation')
                print('\t[%] Load conversation')
                print('\t[^-] Remove conversation')
                print('\t[^^] Save conversation to local filesystem')
                print('\t[%%] Load conversation from local filesystem')
                print('\t[^^-] Remove conversation from local filesystem')
                print('\t')
                print('\t[=] Switch between full and shortened view')
                print('\t[&] Re-generate last GPT message')
                print('\t[] Reload conversation')
                print('\t[\\] Override command')
                print('\t[_] Multiline (Ctrl+X with Enter to exit, Ctrl+C to cancel)' + colorama.Style.RESET_ALL)
                continue

            elif prompt == '??':
                api_key_provided_by = "`api_key` variable" if (api_key != None) else \
                    ("OPENAI_API_KEY environment variable" if ('OPENAI_API_KEY' in os.environ) else None)
                
                print(colorama.Fore.LIGHTGREEN_EX + 'ezgpt is a Python3 library designed to give the user easy and intuitive usage to OpenAI\'s API. It builds on top of OpenAI\'s official `openai` library.')
                print()
                print('Variable information:')
                print(f'\tVersion: {VERSION}')
                print("OpenAI key not provided" if api_key_provided_by == None else f'\tOpenAI key provided using: {api_key_provided_by}')
                print(f'\tImported library `pyperclip`: {has_imported_pyperclip}')
                print(f'\tPersistent conversations get saved in: {PERSISTENT_CONVERSATION_PATH}')
                print(colorama.Fore.BLACK + '\tEaster Egg: Found' + colorama.Fore.LIGHTGREEN_EX)
                print('Repository information:')
                print('\tURL: https://github.com/Ascyt/ezgpt')
                print('\tPyPI URL: https://pypi.org/project/ezgpt')
                print('\tLicense: MIT License')
                print()
                print('About the owner:')
                print('\tHey there! "Ascyt" is my go-to username everywhere. I\'m a 16 year old student and I\'ve been working on this project by myself since November 2023.')
                print('\tMy website: https://ascyt.com/')
                print('\t\u2665 Donate: https://ascyt.com/donate' + colorama.Style.RESET_ALL)
                continue

            elif prompt == '?#':
                print(colorama.Fore.LIGHTGREEN_EX + '(#) Command help:')
                print('\t[model] / [m] The model to generate the completion')
                print('\t[temperature] / [temp] Controls randomness (0-2)')
                print('\t[max_tokens] / [max] The maximum number of tokens to generate')
                print('\t[top_p] / [top] Controls diversity via nucleus sampling (0-1)')
                print('\t[frequency_penalty] / [fp] Decreases token repetition (0-2)')
                print('\t[presence_penalty] / [pp] Increases likelihood of new topics (0-2)' + colorama.Style.RESET_ALL)
                continue
            elif prompt == '#':
                values = (('model', conv.model), ('temperature', conv.temperature), ('max_tokens', conv.max_tokens), ('top_p', conv.top_p), ('frequency_penalty', conv.frequency_penalty), ('presence_penalty', conv.presence_penalty))
                print(colorama.Fore.LIGHTGREEN_EX + '(#) Properties:')
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
                    if prompt == '$ _':
                        prompt = '$_'

                    message = prompt[1:] if len(prompt) > 1 else None

                    if prompt == '$_':
                        message = _get_multiline('system')
                        if message == None:
                            continue

                    conv.system = message

                    reprint_conversation()
                    continue

                elif prompt[0] == '+':
                    if prompt == '+_':
                        prompt = '+ _'
                    if prompt == '++_':
                        prompt = '++ _'

                    space = prompt.find(' ')
                    is_assistant = len(prompt) > 1 and prompt[1] == '+'
                    arg = prompt[space+1:]
                    valueString = prompt[(2 if is_assistant else 1):space]

                    if arg == '_':
                        arg = _get_multiline('assistant' if is_assistant else 'user')
                        if arg == None:
                            continue

                    try:
                        value = int(valueString) if valueString != '' else len(conv.previous)
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

                    valueString = prompt[1:]

                    try:
                        value = int(valueString) if valueString != '' else len(conv.previous) - 1

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

                    if arg == '_':
                        arg = _get_multiline(new_role)
                        if arg == None:
                            continue

                    conv.previous[value] = { 'role': new_role, 'content': arg }
                    reprint_conversation()
                    continue

                elif prompt == '_': 
                    prompt = _get_multiline('user')
                    if prompt == None:
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
                            _print_error(f'`{prompt[1:]}` is not a valid index')
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
                    from_persistent = len(prompt) > 1 and prompt[1] == '%' 

                    arg = prompt[(2 if from_persistent else 1):]

                    conversation = persistant_saved_conversations if from_persistent else saved_conversations

                    if arg == '':
                        if len(conversation) == 0:
                            print(colorama.Fore.LIGHTGREEN_EX + f'No saved conversations{" in local file system" if from_persistent else ""}' + colorama.Style.RESET_ALL)
                            continue

                        print(colorama.Fore.LIGHTGREEN_EX + f'Saved conversations{" in filesystem" if from_persistent else ""}:')
                        for key in list(conversation):
                            print('\t' + key)
                        print(colorama.Style.RESET_ALL, end='')
                        continue

                    value = conversation.get(arg)

                    if value == None:
                        _print_error(f'Conversation "{arg}" not found{f" in `{PERSISTENT_CONVERSATION_PATH}`" if from_persistent else ""}. Type `%` for a list of saved conversations or `%%` for a list of persistently saved conversations')
                        continue

                    conv.previous = value['messages']
                    conv.model = value['model']
                    conv.system = value['system']
                    conv.temperature = value['temperature']
                    conv.top_p = value['top_p']
                    conv.max_tokens = value['max_tokens']
                    conv.frequency_penalty = value['frequency_penalty']
                    conv.presence_penalty = value['presence_penalty']

                    conversation_name = arg

                    reprint_conversation()
                    continue

                elif prompt[0] == '^':
                    save_persistant = len(prompt) > 1 and prompt[1] == '^' 
                    arg = prompt[(2 if save_persistant else 1):]

                    if conversation_name == None:
                        if arg == '':
                            _print_error('Current conversation is not currently saved under a name')
                            continue
                    
                    delete = False
                    if arg[0] == '-':
                        delete = True
                        arg = arg[1:]

                    if arg == '':
                        arg = conversation_name

                    def save_conversation(conversation):
                        if save_persistant: 
                            save_conversation_persistant(arg, conversation)
                            return
                        saved_conversations.setdefault(arg, conversation)
                    def delete_conversation():
                        if save_persistant:
                            persistant_saved_conversations.pop(arg)
                            update_persistant()
                            return
                        saved_conversations.pop(arg)

                    if delete:
                        if arg == None:
                            _print_error('Current conversation has not been saved')
                            continue

                        try:
                            delete_conversation()
                        except KeyError:
                            _print_error(f'Conversation "{arg}" does not exist')
                            continue
                            
                        _print_info(f'Removed conversation "{arg}"{" from local filesystem" if save_persistant else ""}')
                        conversation_name = None
                        continue
                    
                    save_conversation({'messages': conv.previous.copy(), 'model': conv.model, 'system':conv.system, 'temperature':conv.temperature, 'top_p': conv.top_p, 'max_tokens': conv.max_tokens, 'frequency_penalty':conv.frequency_penalty, 'presence_penalty':conv.presence_penalty})

                    conversation_name = arg

                    _print_info(f'Saved conversation{" in local filesystem" if save_persistant else ""} as "{arg}"')
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
                print(colorama.Fore.RED + 'OpenAI Error:\n\t' + e.message + colorama.Style.RESET_ALL)
                conv.previous = conv.previous[:-1]
                cancel_sending = True
                break
    
        if not cancel_sending:
            _print_message({'role':'assistant','content':response}, (not full_view), len(conv.previous) - 1)

def c(model='gpt-3.5-turbo', system=None, messages=None, temperature=0, top_p=0, max_tokens=2048, frequency_penalty=0, presence_penalty=0):
    conversation(model=model, system=system, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

if __name__ == '__main__':  
    c()