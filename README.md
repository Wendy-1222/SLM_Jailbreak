
# Can Small Language Models Reliably Resist Jailbreak Attacks? A Comprehensive Evaluation
<p align="center">
📄 <a href="https://arxiv.org/abs/2503.06519">Paper</a> | 
💻 <a href="https://github.com/Wendy-1222/SLM_Jailbreak">Code</a> | 
🌐 <a href="https://github.com/Wendy-1222/SLM_Jailbreak_Website">Website</a>
</p>

## What did we introduce?
We develop a framework~(based on [HarmBench](https://arxiv.org/abs/2402.04249)) to systematically assess the vulnerability of SLMs to jailbreak attacks. This framework encompasses 14 risk categories and 9 attack methods, and it covers 15 SLM families consists of 62 SLMs, ensuring a thorough and diverse evaluation.

                        
## What did we find? 
Through systematically evaluation on 62 SLMs from 15 mainstream SLM families against 9 state-of-the-art jailbreak methods, we demonstrate that 51.6% of evaluated SLMs show high susceptibility to jailbreak attacks (ASR > 40%) and 38.7% of them can not even resist direct harmful query (ASR > 50%). Through correlation analysis, we identify that the primary factors behind SLM vulnerabilities lie in the training details (e.g., training dataset and method) rather than model size scaling. Additionally, we evaluate the effectiveness of mainstream defense methods that can be applied to SLMs and find that none of them achieve satisfactory results, exhibiting limited performance and poor generalization across different attack methods. Building upon these findings, we highlight the need for security-by-design approaches in SLM development and provide valuable insights for building more trustworthy SLM ecosystem.

## Structure
The directory structure of our framework is organized as follows:

./SLM_Jailbreak
├── adversarial_training  # the original code of HarmBench (not used in our evaluation)
├── allfiles.txt
├── api_models.py  # API models class (not used in our evaluation)
├── baselines # attack methods, including the SLM chat template
├── configs  # method, SLM and pipeline configs
├── cost_analysis  # the cost analysis results
├── data  # the jailbreak questions, you can use AdvBench Subset~(50), AdvBench~(520) or RedAgent dataset~(14 × 5 = 70)
├── defense_baselines  # defense methods
├── docs  # documents of HarmBench
├── environment.yml # our python environment
├── evaluate_completions.py  # original script of HarmBench
├── eval_utils.py  # original script of HarmBench
├── generate_completions.py  # original script of HarmBench
├── generate_defense_completions.py  # original script of HarmBench
├── generate_test_cases.py # original script of HarmBench
├── LICENSE
├── merge_test_cases.py  # original script of HarmBench
├── multimodalmodels  # original script of HarmBench (not used in our evaluation)
├── my_run_scripts  # the scripts used to run our evaluation
├── my_scripts # the scripts to analyze our evaluation results
├── notebooks # original notbooks of HarmBench (not used in our evaluation)
├── figures_and_tables  # the xlsx files and figures of our evaluation, as well as the ipython script for drawing figures (draw.ipynb)
├── __pycache__
├── README.md
├── results # result of RedAgent dataset exclude question in AdvBench Subset
├── results_full_50 # result of AdvBench Subset
├── results_full_50_PAP_transfer # result using PAP-released prompts
├── results_full_70 # result of RedAgent dataset
└── scripts # originals script of HarmBench

## Reproduce Our Results

To reproduce our results, just follow **three simple steps**:

1. **Set up your environment**  
Create your own environment from the provided `environment.yaml` file:  
```bash
conda env create -f environment.yaml
conda activate your_env_name
```

2. **Download the required models**
Run the following script to download all necessary models:
```bash
chmod +x my_scripts/utils/download_models.sh
my_scripts/utils/download_models.sh
```

3. **Run your own tests**
Edit and run the main script for testing:
```bash
python my_run_scripts/main1.py
```


If you find our project useful, please cite:

```
@article{zhang2025can,
  title={Can Small Language Models Reliably Resist Jailbreak Attacks? A Comprehensive Evaluation},
  author={Zhang, Wenhui and Xu, Huiyu and Wang, Zhibo and He, Zeqing and Zhu, Ziqi and Ren, Kui},
  journal={arXiv preprint arXiv:2503.06519},
  year={2025}
}

@article{mazeika2024harmbench,
  title={Harmbench: A standardized evaluation framework for automated red teaming and robust refusal},
  author={Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and others},
  journal={arXiv preprint arXiv:2402.04249},
  year={2024}
}
```
