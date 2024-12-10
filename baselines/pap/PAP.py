import torch
import numpy as np
from tqdm import tqdm
import random
from ..baseline import SingleBehaviorRedTeamingMethod
import ray
from copy import deepcopy
from .templates import persuasion_taxonomy, one_shot_kd, one_shot_kd_contexual, in_context_persuasion_template, in_context_persuasion_template_contextual
from .language_models import load_indiv_model


class PAP(SingleBehaviorRedTeamingMethod):
    use_ray = True

    def __init__(self,
                attack_model,
                attack_gen_config=dict(max_new_tokens=2048, temperature=1.0),
                top_k_persuasion_taxonomy=5,
                **kwargs,
                ):
        """
        :param attack_model: a dictionary specifying the attack model to use (kwargs for load_indiv_model)
        :param attack_gen_config: hyperparameters for attack model generation
        :param top_k_persuasion_taxonomy: number of persuasion techniques to use, based on the ordering in Figure 7 of https://arxiv.org/abs/2401.06373v2
        """
        
        # ====== Load attack model =========
        self.attack_model = load_indiv_model(**attack_model)
        # ====== Generation config for attack and target model ======
        self.attack_gen_config = attack_gen_config
        
        # ======== PAP template =======
        self.persuasion_templates = [dict(technique=d['ss_technique'], definition=d['ss_definition'], example=d['ss_example']) for d in persuasion_taxonomy[:top_k_persuasion_taxonomy]]
    
    # one-shot
    # def generate_test_cases_single_behavior(self, behavior_dict, verbose=False, **kwargs): 
    #     """
    #     Generates test cases for a single behavior

    #     :param behavior: a dictionary specifying the behavior to generate test cases for
    #     :param verbose: whether to print progress
    #     :return: a test case and logs
    #     """
        
    #     # ========== Behavior meta data ==========
    #     behavior = behavior_dict['Behavior']
    #     context_str = behavior_dict['ContextString']
    #     attack_model = self.attack_model

    #     attack_gen_config = self.attack_gen_config
    #     # === PAP Method variables ===
    #     pap_template = deepcopy(one_shot_kd if not context_str else one_shot_kd_contexual)
    #     persuasion_templates = self.persuasion_templates
    #     test_cases = []

    #     # ==== Generate persuasions with attack models==== #
    #     _inputs = [pap_template.format(behavior=behavior, context=context_str, **t) for t in persuasion_templates]
    #     persuasions = get_generations(attack_model, _inputs, **attack_gen_config)
    #     persuasions = [remove_quotes(p.strip()) for p in persuasions]


    #     # ===== Output Test Case =====
    #     test_cases = []
    #     for persuasion in persuasions:
    #         if context_str:
    #             persuasion = f"{context_str}\n\n---\n\n{persuasion}"
    #         test_cases.append(persuasion)

    #     return test_cases, {}
    
    # ICL
    def generate_test_cases_single_behavior(self, behavior_dict, verbose=False, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: five test cases and logs
        """
        
        # ========== Behavior meta data ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        attack_model = self.attack_model

        attack_gen_config = self.attack_gen_config
        # === PAP Method variables ===
        # persuasion_techniques = ["Evidence-based Persuasion", "Expert Endorsement", "Misrepresentation", "Authority endorsement", "Logical appeal"]
        _inputs = []
        for key, value in in_context_persuasion_template.items():
            _inputs.append(value.format(behavior=behavior, context=context_str))

        # ==== Generate persuasions with attack models==== #
        raw_outputs = get_generations(attack_model, _inputs, **attack_gen_config)
        persuasions = []

        for raw_output in raw_outputs:
            raw_output = raw_output.replace('\_', '_')

            # Try to extract content using different tags
            for tag in [
                "mutated_text_with_same_specific_harmful_or_unlawful_intention: ",
                "Mutated_text_with_same_specific_harmful_or_unlawful_intention: ",
                ":",
                "#"
            ]:
                if tag == ":":
                    persuasion = extract_after_second_colon(raw_output)
                else:
                    persuasion = extract_content(tag, raw_output)

                if persuasion:
                    persuasions.append(persuasion)
                    break
            else:
                print(f"Failed to extract persuasion from: {raw_output}")
                persuasions.append(raw_output)

        # ===== Output Test Case =====
        test_cases = []
        for persuasion in persuasions:
            if context_str:
                persuasion = f"{context_str}\n\n---\n\n{persuasion}"
            test_cases.append(persuasion)

        logs = []  # 主要是为了通过baseline.py line 153的 isinstance(batch_test_cases, list) and (len(batch_test_cases) == len(batch_logs))判断，防止PAP只有一个test_case的情况
        for technique, raw_input, raw_output, prompt in zip(
            in_context_persuasion_template.keys(), _inputs, raw_outputs, persuasions
        ):
            log = {
                'technique': technique,
                'raw_input': raw_input,
                'raw_output': raw_output,
                'prompt': prompt
            }
            logs.append(log)

        return test_cases, logs

# ==== Helper Functions ====
# Helper function to check if the model is a ray actor
# Open-source HF and VLLM models are wrapped as ray actors to effectively allocate gpus
def get_generations(model, inputs, **gen_kwargs):
    if isinstance(model, ray.actor.ActorHandle):
       return ray.get(model.batched_generate.remote(inputs, **gen_kwargs))
    else:
        return model.batched_generate(inputs, **gen_kwargs)
    
def remove_quotes(s):
    if s.startswith('"""') and s.endswith('"""'):
        s = s[3:-3]  # Remove triple quotes
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1] # Remove double quotes
    elif s.startswith("'") and s.endswith("'"):
        s = s[1:-1]  # Remove single quotes
    return s

def extract_content(tag, text):
    # Find the starting position of the tag
    start_idx = text.find(tag)

    # If tag is not found, return None
    if start_idx == -1:
        return None
    
    # Extract the content after the tag
    content_after_tag = text[start_idx+len(tag):].strip()
    end_idx = text.find("#", start_idx + 1)
    return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

def extract_after_second_colon(text):
    """
    Extracts content after the second colon in the text.
    Handles cases where the colon is followed by a newline or spaces.
    """
    parts = text.split(":", 2)  # Split into a maximum of 3 parts
    if len(parts) > 2:
        # Strip leading/trailing whitespace from the extracted content
        return parts[2].strip()
    return None