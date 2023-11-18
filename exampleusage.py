import easygpt

gpt = easygpt.gpt()

print(gpt.get(user="hello gpt :)"))
print(gpt.get(messages=gpt.previous, user='what\'s up?.'))
print(gpt.get(messages=gpt.previous, user='what was my first message?'))