# coding=utf-8
import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)   # 数据集的起始和结束位置
parser.add_argument('--end', type=int, default=10)
parser.add_argument('--index', type=int, default=1)   # 用于区分不同的进程
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0, 1])
parser.add_argument('--outdir', type=str, default='outdir')
parser.add_argument('--modelname', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
parser.add_argument('--dataset', type=str, default="7B_behavior_llama2_c.json")

args = parser.parse_args()
maxlen = 2048
maxT = 50   # 最大搜索迭代次数
minT = 5   # 最小搜索迭代次数
Vt = 0.8   # 值阈值V用于决定何时停止搜索
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import time

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from sentence_transformers import SentenceTransformer


def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)

from nodes import node


def save_dict(dict_input, filename):
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


def find_all_indices(text, substring):
    """
    查找字符串 text 中所有子字符串 substring 出现的索引位置
    """
    indices = []
    start_index = 0
    while True:
        index = text.find(substring, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1
    return indices


with open('f1.txt') as f:
    fschat = f.read()
with open('f2.txt') as f:
    fsred = f.read()
with open('r1.txt') as f:
    redA = f.read()
with open('r2.txt') as f:
    redB = f.read()
outdir = args.outdir
try:
    with open('{}/res_{}.json'.format(outdir, args.index)) as f: # 加载已经处理过的越狱问题，res是result
        res = json.loads(f.read())
    qs = []
    for i in res:
        qs.append(i['question'])
except:
    res = []
    qs = []


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    """
    构建一个用于排名任务的数据集。
    感觉就是从原始的越数据集里面加载出越狱的prompt和得到input_ids
    """
    ds = load_dataset('json', data_files=args.dataset)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))  # 选出对应下标的数据
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        """加载越狱的prompt"""
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for i in range(len(examples['goal'])):
            transcript = examples['goal'][i]
            control = examples['controls'][i]
            text = transcript
            text = text + ' ' + control   # 将越狱问题和control拼接在一起
            tokenized_question = tokenizer(text, truncation=True)
            new_examples["query"].append(text)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds1 = ds1.map(   # 将 preprocess_function 应用于数据集中的每一个批次
        preprocess_function,
        batched=True,
        remove_columns=original_columns1,
    )

    ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 512, batched=False) # 过滤掉那些分词后 input_ids 长度大于或等于 512 的样本

    ds1.set_format(type="torch")

    return ds1


if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists('{}_dicv'.format(outdir)):  
    os.makedirs('{}_dicv'.format(outdir))
modelname = args.modelname
print(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    # load_in_4bit=True,
    load_in_8bit=True,
    # torch_dtype=torch.float16,
    device_map="auto",
)
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
testdataset = build_dataset_rank(tokenizer)
testloader = DataLoader(testdataset, batch_size=1, shuffle=False)


@torch.no_grad()
def genc(s, model, tokenizer):
    '''
    vanilla autoregression
    正常生成，产生完整回复
    '''
    input_ids = tokenizer(s, return_tensors="pt").input_ids
    outs = model.generate(inputs=input_ids.cuda(), max_length=maxlen, use_cache=False)
    outstr = tokenizer.decode(outs[0], skip_special_tokens=True)
    return outstr


model.eval()


@torch.no_grad()
def getv(getoken, model, tokenizer, dic, dicp, maxlen):
    '''
    score through self-evaluation
    首先调用 simg 函数生成模拟文本，并与现有文本进行比较，计算文本的得分（输出一个-1到1之间的分数，-1表示非常有害，1表示无害）。
    如果文本不存在评分字典 dic 中，它会计算两种可能的输出的概率并给出评分
    '''
    text, simgstate = simg(dicp, getoken, model, tokenizer, maxlen)
    inds = find_all_indices(text, 'USER:')
    if len(inds) > 1:
        text = text[:inds[1]]
    text = text[inds[0]:]
    text = text.replace('USER:', 'Human:')
    text = text.replace('ASSISTANT:', '\nAssistant:')
    if text not in dic:  # 如果文本 text 不在评分字典 dic 中，表示该文本还未被评分，需要计算评分
        textA = fsred + '\n\n' + text + '\n' + redA     # 根据few-shot examples进行二分类判断
        textB = fsred + '\n\n' + text + '\n' + redB     # redB打乱了option的顺序
        input_ids = tokenizer(textA, return_tensors="pt").input_ids
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = prob[29909].item()  # prob of 'A'
        p_B = prob[29933].item()  # prob of 'B'
        if p_A > p_B:
            A = 1
        else:
            A = 0
        input_ids = tokenizer(textB, return_tensors="pt").input_ids
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = prob[29909].item()
        p_B = prob[29933].item()
        if p_B > p_A:
            B = 1
        else:
            B = 0
        v = (A + B) / 2
        v = (v - 0.5) * 2   # 归一化，使得评分 v 在 -1 和 1 之间
        dic[text] = v
    else:  # 如果文本已经存在于 dic 中，则直接使用已缓存的评分
        v = dic[text]
    return v, simgstate, len(simgstate) - len(getoken)


@torch.no_grad()
def simg(dicp, orstate, model, tokenizer, maxlen=1280):
    '''
    simulation generation for more accurate self-evaluation
    模拟生成。通过在原始状态 orstate 上进行推理，直到生成文本达到 maxlen 限制，或者一句结束了，或者出现多个user；或者遇到结束符 (eos_token_id)。
    '''
    state = copy.deepcopy(orstate)
    past_key_values = None
    while 1:
        if len(state) > maxlen:
            break
        tmpstr = tokenizer.decode(state, skip_special_tokens=True)
        if tmpstr[-1] == ',' or tmpstr[-1] == '.' or tmpstr[-1] == '?' or tmpstr[-1] == ':' or tmpstr[
            -1] == ';' or tmpstr[-1] == '\n':
            break
        inds = find_all_indices(tmpstr, 'USER:')
        if len(inds) > 1:
            break
        probs, past_key_values = getp(state, model, dicp, topk=-1, return_past_key_values=True,
                                      past_key_values=past_key_values)
        token = int(torch.multinomial(probs, num_samples=1))
        state.append(token)
        if token == tokenizer.eos_token_id:
            break
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    return tmpstr, state


def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    根据输入的温度, 重复惩罚, top-p 和 top-k 参数构建一个 LogitsProcessorList
    该列表用于对模型的输出 logits 进行处理。
    """
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
def getp(state, model, dicp, topk=-1, topp=1.0, temperature=1.0, repetition_penalty=0.0, return_last_logits=False,
         return_past_key_values=False, past_key_values=None):
    '''
    query LLM
    获取给定状态下下一步生成token的概率分布的函数
    dicp 主要用于存储每个生成步骤（即每个状态）对应的最后一个 token 的 logits（生成概率的原始值）。这种缓存机制有助于加速后续生成步骤的计算，避免重复计算相同状态下的 logits。
    '''
    if tuple(state) not in dicp:
        if past_key_values != None:
            input_ids = torch.tensor([[state[-1]]])
            outs = model(input_ids.cuda(), past_key_values=past_key_values)
        else:
            input_ids = torch.tensor([state])
            outs = model(input_ids.cuda())
        logits = outs.logits
        past_key_values = outs.past_key_values
        last_logits = logits[:, -1, :].float().cpu()
        dicp[tuple(state)] = last_logits
    else:
        last_logits = dicp[tuple(state)]
        past_key_values = None

    logits_processor = prepare_logits_processor(
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
def group_getp(state, model, dicp, topk=10, maxnew=10, temperature=2.0):
    '''
    group query LLM
    在生成过程中处理多个候选路径（即生成多个 token)并计算多个候选路径的概率。通常用于并行生成或者生成多个可能的序列。
    产生top-k条序列，并且每条序列最多生成maxnew个token
    '''
    outs = []
    outsset = []
    etmpp = []
    if maxnew == 1:
        p, last_logits = getp(state, model, dicp, topk=topk, return_last_logits=True, temperature=temperature)
        acp = p.cpu().detach().squeeze(0).numpy()
        legal = np.where(acp > 0)[0]
        acp = acp[legal]
        acp = zip(legal, acp)
        for ac, p in acp:
            outs.append(([ac], p))
        return outs, last_logits

    greedytmpstate = copy.deepcopy(state)
    greedytmplog = torch.tensor(0.0)
    greedytmptokens = []
    greedy_past_key_values = None
    for i in range(maxnew):
        greedyprobs, greedy_past_key_values = getp(greedytmpstate, model, dicp, topk=15, return_past_key_values=True,
                                                   past_key_values=greedy_past_key_values, temperature=temperature)
        greedytoken = int(torch.argmax(greedyprobs))  # 选择概率最大的 token
        greedylogp = torch.log(greedyprobs[greedytoken])
        greedytmplog += greedylogp
        greedytmptokens.append(greedytoken)
        greedytmpstate.append(greedytoken)
    outsset.append(greedytmptokens)

    for _ in range(2 * topk - 1):
        tmpstate = copy.deepcopy(state)
        tmplog = torch.tensor(0.0)
        tmptokens = []
        past_key_values = None
        for i in range(maxnew):
            probs, past_key_values = getp(tmpstate, model, dicp, topk=15, return_past_key_values=True,
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


def node2dic(node, state, tokenizer):
    """
    将树状结构中的信息（节点和state）转换为更容易处理的字典格式
    """
    d = {}
    dd = {}
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    for act, node in node.children.items():
        actstr = tokenizer.decode(act, skip_special_tokens=True)
        n = node.n  # 该子节点的计数
        q = node.q  # 该子节点的评分，通常用于衡量该路径的质量或价值
        dd[actstr] = (n, q)
    d[tmpstr] = dd
    return d


def getmaxnew(step):
    '''
    return the length of token set
    '''
    if step == 0:
        return 1
    if step == 1:
        return 2
    if step == 2:
        return 4
    return 10


@torch.no_grad()
def search(root, state, model, tokenizer, dic, dicp, maxlen=1024):
    '''
    inner loop
    执行搜索过程，依据当前的文本状态（state）生成多个候选路径，并评估和选择最优路径。
    dic 和 dicp：字典，分别用于存储生成的文本评分和缓存计算的 logits。
    '''
    state = copy.deepcopy(state)
    cnode = root
    reward = 0  # reward 用于记录当前路径的评分
    action = -1

    while not cnode.isleaf():  
        addflag = cnode.checkadd()   # 判断是否需要为当前节点扩展子节点
        if addflag:
            maxnew = getmaxnew(cnode.step)
            agp = group_getp(state, model, dicp, topk=2, maxnew=maxnew)
            cnode.add(agp)
        action, cnode = cnode.select()   # 从子节点中选择最优路径
        state.extend(action)

    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    inds = find_all_indices(tmpstr, 'USER:')
    # check whether the generation is finished
    if len(state) > maxlen or action == tokenizer.eos_token_id or len(inds) > 1 or tokenizer.eos_token_id in state:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
    else:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
        maxnew = getmaxnew(cnode.step)
        if maxnew == 1:
            gp, egp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)
        else:
            gp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)

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
    cnode.backup(v, embeding_token, tokenizer, encoder, path_n=path_n)


@torch.no_grad()
def gmeval(batch, model, tokenizer):
    '''
    outer loop
    '''
    dic, dicp = {}, {}
    query = batch['query'][0]
    query = fschat + 'USER: ' + query + " ASSISTANT:"   # 这个地方也得改成template

    if query in qs:   # 如果问题已经在 qs 中出现过，跳过这个批次
        return None
    instrcot = query
    input_ids = tokenizer(instrcot, return_tensors="pt").input_ids
    slen = input_ids.shape[1]
    state = input_ids.tolist()[0]  # 将输入的文本转换为 state

    root = node(root=None, parent=None, prior_p=0, step=0)  # 以越狱提示为根节点

    initi = 0
    while 1:
        for i in range(initi, max(maxT, initi + 15)):  # 循环执行 search，不断扩展生成的文本状态
            search(root, state, model, tokenizer, dic, dicp, maxlen=maxlen)
            try:
                bq, bfn = root.get_max_nq_value()  # 获取当前生成路径的两个指标：bq（评分）和 bfn（访问次数）
            except:
                bq, bfn = 0, 0
            if bfn > minT and bq > Vt:
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
            rootd = node2dic(root, state, tokenizer)  # 将根节点的信息（包括生成的状态和子节点）转换为字典形式，并保存到文件中
            save_dict(rootd, '{}_dicv/res_root_{}.json'.format(outdir, args.index))

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
            move = tokenizer.eos_token_id

        tmpstr = tokenizer.decode(state, skip_special_tokens=True)
        inds = find_all_indices(tmpstr, 'USER:')
        if len(inds) > 1:
            break
        if len(state) > maxlen:
            break
        if tokenizer.eos_token_id in state:
            break
        if move == tokenizer.eos_token_id:
            break

    raina = tokenizer.decode(state, skip_special_tokens=True)
    inds = find_all_indices(raina, 'USER:')
    if len(inds) > 1:
        raina = raina[:inds[1]]
    raina = raina[inds[0]:]   # rain产生的安全回复

    pa = genc(query, model, tokenizer)   # 模型直接生成的回复
    inds = find_all_indices(pa, 'USER:')
    if len(inds) > 1:
        pa = pa[:inds[1]]
    pa = pa[inds[0]:]

    tmp = {'question': query, 'raina': raina, 'pa': pa}
    save_dict(dic, '{}_dicv/res_{}.json'.format(outdir, args.index))
    return tmp


for epoch_test, batch_test in tqdm(enumerate(testloader)):
    tmp = gmeval(batch_test, model, tokenizer)
    if tmp != None:
        res.append(tmp)
    with open('{}/res_{}.json'.format(outdir, args.index), 'w') as f:
        f.write(json.dumps(res))