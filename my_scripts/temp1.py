import os

metric_list = ['repetition_rate', 'distinct_3', 'entropy_3', 'average_sentence_length', 'lexical_diversity',
               'word_nums', 'sentence_nums', 'self_bleu', 'perplexity', 'readability_score', 'coherence_score']

for metric in metric_list:
    os.system(f"python cal_quality.py --metric {metric}")
    os.system(f"python cal_quality_jailbreak.py --metric {metric}")

