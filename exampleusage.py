from ezgpt import ezgpt
import asyncio

prompt = 'give python factorial number'

gpt = ezgpt.gpt(logs=True)

print('Improve prompt...')
improved:str = asyncio.run(gpt.get(user=prompt, system="You will be provided with a prompt. Your task is not to implement it, but to correct it and improve its phrasing."))

print('Implement prompt...')
implemented:str = asyncio.run(gpt.get(user=improved + " Keep yourself short.", messages=[]))
print('Generate explanation...')
explanation:str = asyncio.run(gpt.get(messages=gpt.previous, user="Now explain this code in a simple and short way."))

print(f'Implementation:\n\n{implemented}')
print(f'Explanation:\n\n{explanation}')