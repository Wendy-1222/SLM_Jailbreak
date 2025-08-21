import os

classifier_list = ['Llama-Guard-3-1B', 'Llama-Guard-3-8B', 'roberta']
# classifier_list = ['llama3_1_8b']

for classifier in classifier_list:
    os.system(f'python generate_full_70_test_cases.py --classifier {classifier}')
