import os

# defender_list = ['ppl', 'self-reminder', 'retokenization', 'llama_guard_3', 'llama_guard_3_1B']
# defender_list = ['Llama-Guard-3-1B_output_filter', 'Llama-Guard-3-8B_output_filter']
defender_list = ['self-reminder-with-system-prompt', 'self-reminder-only-system-prompt']
for defender in defender_list:
    # os.system(f'python generate_full_70_test_cases.py --defender {defender}')
    os.system(f'python count_score_defense.py --defender {defender}')