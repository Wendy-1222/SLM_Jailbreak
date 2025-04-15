import os

defender_list = ['ppl', 'self-reminder', 'retokenization', 'llama_guard_3', 'llama_guard_3_1B']
for defender in defender_list:
    # os.system(f'python generate_full_70_test_cases.py --defender {defender}')
    os.system(f'python count_score_defense.py --defender {defender}')