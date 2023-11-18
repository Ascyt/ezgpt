import easygpt

prompt = 'give python factorial number'

gpt = easygpt.gpt(logs=True)

print('Improve prompt...')
improved:str = gpt.get(user=prompt, system="You will be provided with a prompt. Your task is not to implement it, but to correct it and improve its phrasing.")

print('Implement prompt...')
implemented:str = gpt.get(user=improved + " Keep yourself short.", messages=[])
print('Generate explanation...')
explanation:str = gpt.get(messages=gpt.previous, user="Now explain this code in a simple and short way.")

print(f'Implementation:\n\n{implemented}')
print(f'Explanation:\n\n{explanation}')