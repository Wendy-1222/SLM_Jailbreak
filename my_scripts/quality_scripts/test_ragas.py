# #需要openai的api key
# from ragas import evaluate
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_recall,
#     context_precision,
# )

# questions = ["What is the capital of France?"]
# answers = ["Paris"]
# contexts = ["France is a country in Europe. The capital of France is Paris."]
# ground_truths = ["Paris"]

# # To dict
# data = {
#     "question": questions,
#     "answer": answers,
#     "contexts": contexts,
#     "ground_truths": ground_truths
# }

# result = evaluate(
#     dataset = data, 
#     metrics=[
#         context_precision,
#         context_recall,
#         faithfulness,
#         answer_relevancy,
#     ],
# )

# print(result)