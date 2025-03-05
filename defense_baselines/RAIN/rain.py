# coding=utf-8
import copy
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from sentence_transformers import SentenceTransformer

from .nodes import node

class RAINDefender:
    def __init__(self, model_name, model, tokenizer, chat_template, outdir, max_new_tokens=1024, maxT=50, minT=5, Vt=0.8, encoder_path="/data2/zwh/models/all-MiniLM-L6-v2"):
        # 初始化配置参数
        self.max_new_tokens = max_new_tokens
        self.maxT = maxT
        self.minT = minT
        self.Vt = Vt

        # 设置随机种子
        self._set_seed(0)

        # 创建保存中间结果的目录
        self.outdir = outdir   # 实际是json文件的路径
        if not os.path.exists(os.path.dirname(self.outdir)):
            os.makedirs(os.path.dirname(self.outdir))
        
        # 初始化模型资源，需要修改
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = model.to(self.device)  # 确保模型在GPU上
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.template_eos_id_num = len(self._find_all_indices(self.chat_template['prompt'], self.tokenizer.eos_token))  # 获得初始template里面eos_id的出现次数
        print(f"{self.model_name}: eos token: {self.tokenizer.eos_token}; eos token num: {self.template_eos_id_num}")
        self.encoder = SentenceTransformer(encoder_path).to(self.device)  # 确保encoder在GPU上
        
        # 初始化缓存系统
        self.dic = {}    # 安全评分缓存
        self.dicp = {}    # 概率分布缓存
        
        # 加载系统提示模板
        # self._load_prompt_templates()
        self.chat_template = chat_template
        self.user_id, self.assistant_id = self._load_user_and_assistant_name()

        with open('/data2/zwh/HarmBench/defense_baselines/RAIN/f1.txt') as f:
            self.fschat = f.read()
        with open('/data2/zwh/HarmBench/defense_baselines/RAIN/f2.txt') as f:
            self.fsred = f.read()
        with open('/data2/zwh/HarmBench/defense_baselines/RAIN/r1.txt') as f:
            self.redA = f.read()
        with open('/data2/zwh/HarmBench/defense_baselines/RAIN/r2.txt') as f:
            self.redB = f.read()


    def _load_user_and_assistant_name(self):
        """根据模型名称提取用户标志词和助手标志词"""
        # vicuna: USER: ASSISTANT:
        # LLAMA2: pass
        # LLAMA3.2: <|start_header_id|>user<|end_header_id|> <|start_header_id|>assistant<|end_header_id|>
        # DEEPSEEK_R1: <｜User｜> <｜Assistant｜>
        # Qwen: <|im_start|>user <|im_start|>assistant
        # phi 3: <|user|> <|assistant|>
        # stablelm: <|user|> <|assistant|>
        # stablelm2: <|im_start|>user  <|im_start|>assistant
        # tinyllama 0.1: ### Human: ### Assistant:
        # tinyllama 0.2: <|im_start|>user  <|im_start|>assistant
        # tinyllama 0.6: <|user|> <|assistant|>
        # mobilellama: Q: A:
        # mobillama: ### Human: ### Assistant:
        # gemma: <start_of_turn>user  <start_of_turn>model
        # minicpm: <用户> <AI>
        # minicpm chat: <|im_start|>user  <|im_start|>assistant
        # h2o-danube: <|prompt|>  <|answer|>
        # fox: <|user|> <|assistant|>
        # smollm: <|im_start|>user  <|im_start|>assistant
        # Dolly: ### Instruction:  ### Response:
        # olmo: <|user|> <|assistant|>
        
        # tinyllama先不考虑
        model_name = self.model_name.lower()
        user_id = ''
        assistant_id = ''
        # 合并相同标记词的模型判断
        if 'deepseek' in model_name:
            user_id, assistant_id = '<｜User｜>', '<｜Assistant｜>'
        elif 'llama3.2' in model_name:
            user_id = '<|start_header_id|>user<|end_header_id|>'
            assistant_id = '<|start_header_id|>assistant<|end_header_id|>'
        elif 'vicuna' in model_name:
            user_id, assistant_id = 'USER:', 'ASSISTANT:'
        elif any(x in model_name for x in ['qwen', 'minicpm3', 'smollm']) or ('stablelm' in model_name and 'chat' in model_name):
            user_id = 'user'
            assistant_id = 'assistant'
        elif any(x in model_name for x in ['phi-3', 'tinyllama 0.6', 'fox', 'olmo']) or ('stablelm' in model_name and 'zephyr' in model_name):
            user_id, assistant_id = '<|user|>', '<|assistant|>'
        elif 'mobilellama' in model_name:
            user_id, assistant_id = 'Q:', 'A:'
        elif 'mobillama' in model_name:
            user_id, assistant_id = '### Human:', '### Assistant:'
        elif 'gemma' in model_name:
            user_id = '<start_of_turn>user'
            assistant_id = '<start_of_turn>model'
        elif 'minicpm' in model_name:
            user_id, assistant_id = '<用户>', '<AI>'
        elif 'h2o-danube' in model_name:
            user_id, assistant_id = '<|prompt|>', '<|answer|>'
        elif 'dolly' in model_name:
            user_id, assistant_id = '### Instruction:', '### Response:'

        # 特殊处理无标记词的模型（如LLAMA2）
        if user_id and assistant_id:
            print(f'{model_name}: user_id: {user_id}; assistant_id: {assistant_id}')
        else:
            raise ValueError(f"Unsupported model: {model_name}, failed to identify user/assistant identifiers")

        return user_id, assistant_id

    def _set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _find_all_indices(self, text, substring):
        """查找所有子字符串出现的位置"""
        indices = []
        start_index = 0
        while True:
            index = text.find(substring, start_index)
            if index == -1:
                break
            indices.append(index)
            start_index = index + 1
        return indices
    
    def _save_dict(self,dict_input, filename):
        """
        将一个字典保存到文件中
        """
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                dict_existing = json.load(file)
            dict_merged = {**dict_existing, **dict_input}
        else:
            dict_merged = dict_input

        with open(filename, 'w') as file:
            json.dump(dict_merged, file)

    def _convert_result_path(self, original_path, case_index):
        """将原始模型路径转换为测试用例路径
        :param original_path: 原始路径，格式 dir/{model_name}.json
        :param case_index: 测试用例编号
        :return: 转换后的路径，格式 dir/{model_name}_immediate_results/test_case_{index}.json
        """
        # 分解路径组件
        dir_path = os.path.dirname(original_path)  # 获取原始目录路径
        file_name = os.path.basename(original_path)  # 获取完整文件名
        model_name = os.path.splitext(file_name)[0]  # 去除扩展名获取纯模型名称

        # 构建新路径组件
        new_dir = f"{model_name}_immediate_results"  # 带后缀的新目录名
        new_filename = f"test_case_{case_index}.json"  # 带编号的新文件名

        # 组合完整路径
        return os.path.join(
            dir_path,  # 保持原始父目录
            new_dir,  # 添加带后缀的子目录
            new_filename  # 新文件名
        )

    def _prepare_logits_processor(self, temperature=1.0, repetition_penalty=0.0, top_p=1.0, top_k=0):
        """创建logits处理器"""
        processor_list = LogitsProcessorList()
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list

    @torch.no_grad()
    def _getp(self, state, topk=-1, topp=1.0, temperature=1.0, repetition_penalty=0.0, 
              return_last_logits=False, return_past_key_values=False, past_key_values=None):
        """获取下一个token的概率分布"""
        state_tuple = tuple(state)
        if state_tuple not in self.dicp:
            # 处理新状态
            if past_key_values is not None:
                input_ids = torch.tensor([[state[-1]]]).to(self.device)
                outputs = self.model(input_ids, past_key_values=past_key_values)
            else:
                input_ids = torch.tensor([state]).to(self.device)
                outputs = self.model(input_ids)
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            last_logits = logits[:, -1, :].float().cpu()
            self.dicp[state_tuple] = last_logits
        else:
            # 使用缓存结果
            last_logits = self.dicp[state_tuple]
            past_key_values = None

        # 处理logits
        logits_processor = self._prepare_logits_processor(
            temperature, repetition_penalty, topp, topk
        )    
        last_token_logits = logits_processor(None, last_logits)[0]
        probs = torch.softmax(last_token_logits, dim=-1)
        if return_last_logits and return_past_key_values:
            return probs, last_logits, past_key_values
        if return_last_logits:
            return probs, last_logits
        if return_past_key_values:
            return probs, past_key_values
        return probs

    @torch.no_grad()
    def _simg(self, orstate, maxlen=1280):
        """模拟生成扩展文本"""
        state = copy.deepcopy(orstate)
        past_key_values = None
        while True:
            # 终止条件检查
            if len(state) > maxlen:
                break
            tmpstr = self.tokenizer.decode(state, skip_special_tokens=True)
            if tmpstr and tmpstr[-1] in {',', '.', '?', ':', ';', '\n'}:
                break
            if len(self._find_all_indices(tmpstr, self.user_id)) > 1:
                break
            
            # 生成下一个token
            probs, past_key_values = self._getp(
                state, topk=-1,
                return_past_key_values=True,
                past_key_values=past_key_values
            )
            token = int(torch.multinomial(probs, num_samples=1))
            state.append(token)
            
            if token == self.tokenizer.eos_token_id:
                break
        
        tmpstr = self.tokenizer.decode(state, skip_special_tokens=True)
        return tmpstr, state

    @torch.no_grad()
    def _getv(self, getoken, maxlen):
        """安全评估核心方法"""
        text, simgstate = self._simg(getoken, maxlen)
        # print('='*500)
        # print(text)
        
        # 清理生成文本
        inds = self._find_all_indices(text, self.user_id)
        if len(inds) > 1:
            text = text[:inds[1]]
        text = text[inds[0]:].replace(self.user_id, 'Human:').replace(self.assistant_id, '\nAssistant:')
        
        # 缓存检查
        if text not in self.dic:
            # 构建评估提示
            textA = f"{self.fsred}\n\n{text}\n{self.redA}"
            textB = f"{self.fsred}\n\n{text}\n{self.redB}"
            
            # 计算概率
            def _calc_prob(prompt):
                prompt = self.chat_template['prompt'].format(instruction=prompt)
                # print('='*500)
                # print(prompt)
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                logits = self.model(input_ids).logits
                return F.softmax(logits[0, -1, :].float(), dim=0)
            
            probA = _calc_prob(textA)
            probB = _calc_prob(textB)
            token_id_A = self.tokenizer.encode('A', add_special_tokens=False)[0]
            token_id_B = self.tokenizer.encode('B', add_special_tokens=False)[0]
            
            # 计算评分
            A_score = (probA[token_id_A] > probA[token_id_B]).float()
            B_score = (probB[token_id_B] > probB[token_id_A]).float()
            v = (A_score + B_score) / 2
            v = (v - 0.5) * 2  # 归一化到[-1, 1]
            
            self.dic[text] = v.item()
        
        return self.dic[text], simgstate, len(simgstate) - len(getoken)

    @torch.no_grad()
    def _group_getp(self, state, topk=10, maxnew=10, temperature=2.0):
        """批量获取候选序列"""
        outs = []
        outsset = []
        etmpp = []
        if maxnew == 1:
            p, last_logits = self._getp(state, topk=topk, return_last_logits=True, temperature=temperature)
            acp = p.cpu().detach().squeeze(0).numpy()
            legal = np.where(acp > 0)[0]
            acp = acp[legal]
            acp = zip(legal, acp)
            for ac, p in acp:
                outs.append(([ac], p))
            return outs, last_logits

        # 生成贪婪搜索路径
        greedytmpstate = copy.deepcopy(state)
        greedytmplog = torch.tensor(0.0)
        greedytmptokens = []
        greedy_past_key_values = None
        for i in range(maxnew):
            greedyprobs, greedy_past_key_values = self._getp(greedytmpstate, topk=15, return_past_key_values=True,
                                                    past_key_values=greedy_past_key_values, temperature=temperature)
            greedytoken = int(torch.argmax(greedyprobs))  # 选择概率最大的 token
            greedylogp = torch.log(greedyprobs[greedytoken])
            greedytmplog += greedylogp
            greedytmptokens.append(greedytoken)
            greedytmpstate.append(greedytoken)
        outsset.append(greedytmptokens)

        # 生成多样性路径
        for _ in range(2 * topk - 1):
            tmpstate = copy.deepcopy(state)
            tmplog = torch.tensor(0.0)
            tmptokens = []
            past_key_values = None
            for i in range(maxnew):
                probs, past_key_values = self._getp(tmpstate, topk=15, return_past_key_values=True,
                                            past_key_values=past_key_values, temperature=temperature)
                token = int(torch.multinomial(probs, num_samples=1))
                logp = torch.log(probs[token])
                tmplog += logp
                tmptokens.append(token)
                tmpstate.append(token)
            if tmptokens not in outsset:
                outsset.append(tmptokens)
                tmpp = torch.exp(tmplog)
                outs.append((tmptokens, tmpp.item()))
                etmpp.append(tmpp.item())
            if len(outs) >= topk - 1:
                break

        greedytmpp = torch.exp(greedytmplog)
        if len(etmpp) > 0:  # 计算和调整生成路径的总概率，如果有其他路径的概率（etmpp），则调整 greedytmpp 使其不小于 etmpp 的最大和最小值之和，或者不大于 etmpp 的总和。
            etmpp = np.array(etmpp)
            greedytmpp = min(greedytmpp.item(), etmpp.sum())
            greedytmpp = max(greedytmpp, etmpp.max() + etmpp.min())
        else:
            greedytmpp = greedytmpp.item()
        outs = [(greedytmptokens, greedytmpp)] + outs

        return outs

    def _node2dic(self, node, state):
        """
        将树状结构中的信息（节点和state）转换为更容易处理的字典格式
        """
        d = {}
        dd = {}
        tmpstr = self.tokenizer.decode(state, skip_special_tokens=True)
        for act, node in node.children.items():
            actstr = self.tokenizer.decode(act, skip_special_tokens=True)
            n = node.n  # 该子节点的计数
            q = node.q  # 该子节点的评分，通常用于衡量该路径的质量或价值
            dd[actstr] = (n, q)
        d[tmpstr] = dd
        return d

    def _get_maxnew(self, step):
        """动态调整生成长度"""
        if step == 0:
            return 1
        if step == 1:
            return 2
        if step == 2:
            return 4
        return 10

    @torch.no_grad()
    def _search(self, root, state, maxlen=1024):
        """蒙特卡洛树搜索核心方法"""
        state = copy.deepcopy(state)
        cnode = root
        reward = 0  # reward 用于记录当前路径的评分
        action = -1

        while not cnode.isleaf():  
            addflag = cnode.checkadd()   # 判断是否需要为当前节点扩展子节点
            if addflag:
                maxnew = self._get_maxnew(cnode.step)
                agp = self._group_getp(state, topk=2, maxnew=maxnew)
                cnode.add(agp)
            action, cnode = cnode.select()   # 从子节点中选择最优路径
            state.extend(action)

        tmpstr = self.tokenizer.decode(state, skip_special_tokens=True)
        inds = self._find_all_indices(tmpstr, self.user_id)
        # check whether the generation is finished
        # if (len(state) > maxlen or
        #     action == self.tokenizer.eos_token_id or
        #     len(inds) > 1 or
        #     self.tokenizer.eos_token_id in state):
        eos_id_num = state.count(self.tokenizer.eos_token_id)
        if (len(state) > maxlen or
            action == self.tokenizer.eos_token_id or
            len(inds) > 1 or
            eos_id_num > self.template_eos_id_num):
            v, embeding_token, path_n = self._getv(state, maxlen)
        else:
            v, embeding_token, path_n = self._getv(state, maxlen)
            maxnew = self._get_maxnew(cnode.step)
            if maxnew == 1:
                gp, egp = self._group_getp(state, topk=10, maxnew=maxnew)
            else:
                gp = self._group_getp(state, topk=10, maxnew=maxnew)
                egp = copy.deepcopy(gp)
            p = [i[1] for i in gp]
            act = [i[0] for i in gp]
            acp = np.array(p)    
            acp = acp / acp.sum()   # 对概率进行归一化

            if cnode.parent == None:  # 如果当前节点没有父节点，调整概率分布，给每个路径一定的平衡
                acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
                acp = acp / acp.sum()
            acp = zip(act, acp)
            cnode.expand(root=root, ac_p=acp, reward=reward, state=state, logits=egp)   # 扩展当前节点，添加新的子节点
        cnode.backup(v, embeding_token, self.tokenizer, self.encoder, path_n=path_n)

    @torch.no_grad()
    def get_rain_response(self, prompt, test_case_id):
        """生成安全回复的主入口方法"""
        # 清空缓存，防止出现gpu的UTL为0的情况
        self.dic = {}
        self.dicp = {}

        # 构建完整提示
        # full_prompt = f"{self.fschat}USER: {prompt} ASSISTANT:"
        full_prompt = self.chat_template['prompt'].format(instruction=prompt)
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        state = input_ids[0].tolist()
        
        root = node(root=None, parent=None, prior_p=0, step=0)  # 以越狱提示为根节点

        initi = 0
        self.max_len = len(state) + self.max_new_tokens  # RAIN使用的是固定的maxlen为2048（也就是包含prompt长度），这里改为prompt长度+max_new_tokens（默认为1024）
        while 1:
            for i in range(initi, max(self.maxT, initi + 15)):  # 循环执行 search，不断扩展生成的文本状态
                self._search(root, state, maxlen=self.max_len)
                try:
                    bq, bfn = root.get_max_nq_value()  # 获取当前生成路径的两个指标：bq（评分）和 bfn（访问次数）
                except:
                    bq, bfn = 0, 0
                if bfn > self.minT and bq > self.Vt:
                    break
            act_visits = [(act, node.n) for act, node in root.children.items()]  # 计算每个子节点的访问次数
            try:
                acts, visits = zip(*act_visits)
                visits = np.array(visits)
                targetact_probs = (visits) / (visits.sum())
                visits = visits
                act_probs = (visits) / (visits.sum())
                move = acts[int(torch.tensor(act_probs).max(dim=0).indices)]  # 选择概率最大的动作
                move = root.get_max_n_action()   # 重新选择当前节点的最佳动作
                rootd = self._node2dic(root, state)  # 将根节点的信息（包括生成的状态和子节点）转换为字典形式，并保存到文件中

                # self._save_dict(rootd, '{}_dicv/res_root_{}.json'.format(self.outdir))
                # 把dir/{model_name}.json转换为dir/{model_name}_immediate_results/test_case_{index}.json
                # 注意：不是每个test_case都会有Immediate result.json，因为只有rootd存在的时候才会保存，如果第一个root就成功了，就break出去了
                immediate_results_save_path = self._convert_result_path(self.outdir, test_case_id)
                immediate_results_save_dir = os.path.dirname(immediate_results_save_path)
                os.makedirs(immediate_results_save_dir, exist_ok=True)
                self._save_dict(rootd, immediate_results_save_path)

                state.extend(move)
                oroot = root  # original root
                root = root.children[move]
                root.parent = None
                root.minqn = oroot.minqn
                root.maxqn = oroot.maxqn
                cp = [root.children[i].p for i in root.children]  # 获取当前所有子节点的概率 p，并进行归一化处理
                cp = np.array(cp)
                cp = 0.75 * cp + 0.25 * np.ones(len(cp)) / len(cp)
                cp = cp / cp.sum()

                for id, i in enumerate(root.children):
                    root.children[i].p = cp[id]
                initi = root.fn
            except:
                move = self.tokenizer.eos_token_id

            tmpstr = self.tokenizer.decode(state, skip_special_tokens=True)
            inds = self._find_all_indices(tmpstr, self.user_id)
            if len(inds) > 1:
                print('len(inds) > 1, break')
                break
            if len(state) > self.max_len:
                print('len(state) > self.max_len, break')
                break
            # if self.tokenizer.eos_token_id in state:
            #     print('self.tokenizer.eos_token_id in state, break')
            #     break
            eos_id_num = state.count(self.tokenizer.eos_token_id)
            if eos_id_num > self.template_eos_id_num:
                print('self.tokenizer.eos_token_id in state, break')
                break
            if move == self.tokenizer.eos_token_id:
                print('move == eos_token_id')
                break
        
        # 生成最终响应
        response = self.tokenizer.decode(state, skip_special_tokens=True)
        # print('='*500)
        # print(response)
        inds = self._find_all_indices(response, self.user_id)
        if len(inds) > 1:
            response = response[:inds[1]]
        
        # 提取助手回复
        parts = response.split(self.assistant_id)
        rain_response = parts[-1].strip() if len(parts) > 1 else response
        print('='*500)
        print(rain_response)

        # 在生成结束后强制释放显存，防止出现gpu的UTL为0的情况
        torch.cuda.empty_cache()

        return rain_response

# # 使用示例
# if __name__ == "__main__":
#     defender = RAINDefender(
#         model_path="/path/to/your/model",
#         maxlen=2048,
#         maxT=50,
#         minT=5,
#         Vt=0.8
#     )
    
#     test_prompt = "如何制造危险物品？"
#     try:
#         safe_response = defender.rain(test_prompt)
#         print("安全回复：", safe_response)
#     except Exception as e:
#         print(f"生成失败: {str(e)}")