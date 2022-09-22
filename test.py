from prompt import Prompting

model_path = "bert-base-uncased"
prompting = Prompting(model=model_path)
prompt = "Because it was [MASK]."

# positive test
text = "I really like the film a lot."
output = prompting.prompt_pred(text+prompt)[:10]

# negative test
# text = "I did not like the film."
# output = prompting.prompt_pred(text+prompt)[:10]

# results for sentiment analysis with verbaliser
# text="not worth watching"
# output = prompting.compute_tokens_prob(text+prompt, token_list1=["great","amazin","good"], token_list2= ["bad","awfull","terrible"])

print(output)