from TinyAgent import LLM

# Gemma 2 12B (no native thinking or tool calling)
llm = LLM(model="gemma3:12b")

def llm_response(llm: LLM, prompt_example: str, prompt_answer: str, query: str):
# Messages
    messages = [
        {"role": "user", "content": prompt_example},
        {"role": "assistant", "content": prompt_answer},
        {"role": "user", "content": query}
    ]
    response = llm.generate(messages)

    return response


# Example of one-shot prompting
prompt_example = "I saw 6 flamingos. 2 flew away. 1 hidden behind a tree. How many can I see? Be concise."
prompt_answer = "3"

# The query we want to ask the model
query = "I saw 9 penguins. 2 slid in the water and disappeared from sight while 4 waddled up from the shore. How many can I see?"



# Generate response
response = llm_response(llm, prompt_example, prompt_answer, query)
print(response)

prompt_example = "I saw 6 flamingos. 2 flew away. 1 hidden behind a tree. How many can I see?"
prompt_answer = "You saw 6 flamingos 2 flew away. 6 - 2 = 4. Then, 1 flamingo hid. you see now 4 - 1 = 3."

# Generate response
response = llm_response(llm, prompt_example, prompt_answer, query)
print(response)

# zero-shot prompting
messages = [{"role": "user", "content": query + " Let's think step by step."}]

# Generate response
response = llm.generate(messages)
print(response)