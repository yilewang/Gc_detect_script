import openai


openai_api_key_filename = '/Users/yat-lok/Documents/openai_key.txt'
with open(openai_api_key_filename, 'r') as f:
    openai.api_key = f.read()

# print(openai.api_key)  # sanity check
openai.api_key = openai.api_key.strip('\n')  # remove trailing newline

message_history = []
def message_rollup(inp, role):
    message_history.append({"role":role, "content":inp})
    response = openai.ChatCompletion.create(model = "gpt-3.5-turbo", 
                                            messages = message_history)
    gpt_reply = response['choices'][0].message.content
    message_history.append({"role":"assistant", "content":gpt_reply})
    return gpt_reply

n=1
i=0
while i < n:
    inp = input("You: \n")
    if inp.lower() == 'stop':
        break
    print("GPT-3.5-turbo: \n", message_rollup(inp, "user"))