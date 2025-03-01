"""
This code is based on the implementation of TSA.
Reference: https://github.com/VICO-UoE/URL
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import random
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from models.losses import prototype_loss
from models.ta import ta
from data.meta_dataset_reader import MetaDatasetEpisodeReader
from config import args

# stronger models
from models.resnet50 import create_model
from copy import deepcopy
import torch.nn as nn
import clip
from deit.models import deit_small_patch16_224
from swin_transformer.models.swin_transformer import SwinTransformer
from torchvision import transforms
import torchvision.transforms as T

def preserve_key(state, remove_prefix: str):
    """Preserve part of model weights based on the
       prefix of the preserved module name.
    """
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if remove_prefix + '.' in key:
            newkey = key.replace(remove_prefix + '.', "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    return state

def norm_clip(imgs):
    """将图像归一化到CLIP模型的输入格式""" 
    transform_img = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711]))])
    imgs_list = []
    for img in imgs:
        # img = img.squeeze(0)
        # 将张量转换为PIL图像并进行归一化  
        to_pil = T.ToPILImage() # 转为PIL格式  
        img = (img / 2 + 0.5) * 255 # 反归一化到0-255范围  
        img = img.to(torch.uint8)
        img = to_pil(img)
        imgs_list.append(transform_img(img))   # 应用CLIP预处理 
    img_tensor = torch.stack(imgs_list).cuda() # 堆叠并转移到GPU  

    return img_tensor

ALL_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco".split(' ')
TRAIN_METADATASET_NAMES = ALL_METADATASET_NAMES[:8]
TEST_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco" .split(' ')

def main():
    # 初始化测试和训练数据集
    testsets = TEST_METADATASET_NAMES
    trainsets = TRAIN_METADATASET_NAMES
    # 创建测试数据读取器  
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])

    model_name = args['pretrained_model']
    ratio = args['ratio']

    if args['ours']:
        is_baseline = False
    else:
        is_baseline = True
    K_patch = args['n_regions'] # 图像分块数量 
    max_It = args['maxIt'] # 最大迭代次数 
    TEST_SIZE = 600 # 测试任务数量  

    # 是否启用权重调整  
    is_weight_patch = True
    is_weight_sample = True

    # 根据模型类型设置微调学习率  
    if model_name == 'CLIP':
        lr_finetune = 0.001
    elif model_name == 'MOCO':
        lr_finetune = 0.001
    elif model_name == 'DEIT':
        lr_finetune = 0.1
    elif model_name == 'SWIN':
        lr_finetune = 0.05
    else:
        lr_finetune = 0.

    # RN-50
    if model_name == 'CLIP':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("RN50", device=device)  # 加载CLIP模型
    if model_name == 'MOCO':
        model = create_model() # 创建自定义模型  
        state = torch.load("models/moco_v2_800ep_pretrain.pth.tar")["state_dict"]
        state = preserve_key(state, "module.encoder_q")  # 调整权重键名 
        # 删除全连接层参数  
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "fc" in key:
                state.pop(key)
        model.load_state_dict(state) # 加载调整后的权重  
    # Vit
    if model_name == 'DEIT':
        model = deit_small_patch16_224(pretrained=True)  # 加载预训练DEIT
    if model_name == 'SWIN':
        model = SwinTransformer()
        state = torch.load("models/swin_tiny_patch4_window7_224.pth")['model']
        model.load_state_dict(state)  # 加载Swin权重  

    model.eval() # 设置为评估模式
    model.cuda() # 转移到GPU  
    accs_names = ['NCC']  # 评估指标名称（此处为最近类中心分类）
    var_accs = dict() # 存储各数据集准确率

    # 创建TensorFlow会话（可能用于数据加载）  
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # 遍历所有测试数据集  
        for dataset in testsets:
            var_accs[dataset] = {name: [] for name in accs_names} # 初始化准确率存储

            if is_baseline: # 基线方法关闭权重调整
                is_weight_patch = False
                is_weight_sample = False
            # 对每个测试任务进行推理    
            for i in tqdm(range(TEST_SIZE)):
                model.zero_grad()
                torch.cuda.empty_cache() # 清空GPU缓存

                # 从测试集获取一个任务的数据  
                sample = test_loader.get_test_task(session, dataset)
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']
                context_images = sample['context_images']
                target_images = sample['target_images']

                # Label bias 处理标签偏置（按比例替换支撑集样本）
                if ratio > 0:
                    N_shots = 10 # 每类样本数
                    N_bias = int(N_shots * ratio) # 需替换的样本数
                    data = []
                    N_way = context_labels[-1] + 1 # 类别数 
                    for nw in range(N_way):
                        label_others = []
                        for t in range(N_way):
                            # 随机选择其他类标签  
                            if t != nw:
                                label_others.append(t)
                        random.shuffle(label_others) #打乱列表元素顺序
                        num_others = len(label_others)
                        # 获取当前类别的support set  
                        data_nw = context_images[nw*N_shots:(nw+1)*N_shots]
                        # 替换最后N_bias个样本为其他类样本 
                        for tt in range(N_bias):
                            if num_others < tt+1:
                                ttt = tt+1-num_others
                            else:
                                ttt = tt
                            other_class_label = label_others[ttt]
                            start_others = other_class_label * N_shots
                            randindx = random.randint(0, N_shots-1)
                            index_others = start_others + randindx #替代样本当前任务的index
                            data_nw[N_shots-1-tt] = context_images[index_others] #替换成错误样本
                        data.append(data_nw)
                    context_images = torch.cat(data) # 合并处理后的支撑集
                
                # 深拷贝模型以避免修改原始模型 
                cur_model = deepcopy(model)
                # 根据模型类型调整分块数量（节省内存）
                if (model_name == 'DEIT' and context_labels.shape[0] >90) \
                        or (model_name == 'SWIN' and context_labels.shape[0] >70): # for saving time/memory
                    K_patch = 1
                
                # 调用TA
                sample_weight = ta(context_images, context_labels, cur_model, model_name=model_name, max_iter=max_It, lr_finetune=lr_finetune, distance=args['test.distance'],
                                         is_baseline = is_baseline, is_weight_patch=is_weight_patch, is_weight_sample=is_weight_sample, K_patch=K_patch, dataset=dataset)
                # 提取特征  
                with torch.no_grad():
                    if model_name == 'CLIP': # 使用CLIP编码图像  
                        context_features = cur_model.encode_image(context_images)  # (context_images)##
                        target_features = cur_model.encode_image(target_images)  # (target_images)######
                    else: # 普通前向传播  
                        context_features = cur_model(context_images)
                        target_features = cur_model(target_images)
                    if len(context_features.shape) == 4: # 处理4D特征（如CNN输出）  
                        avg_pool = nn.AvgPool2d(context_features.shape[-2:])
                        context_features = avg_pool(context_features).squeeze(-1).squeeze(-1)
                        target_features = avg_pool(target_features).squeeze(-1).squeeze(-1)
                    if is_weight_sample: # 应用样本权重  
                        context_features = sample_weight.unsqueeze(-1) * context_features

                # 计算原型损失并获取准确率  
                _, stats_dict, _ = prototype_loss(
                    context_features, context_labels,
                    target_features, target_labels, patch_weight=None, distance=args['test.distance'])
                var_accs[dataset]['NCC'].append(stats_dict['acc']) # 记录准确率

            # 输出当前数据集平均准确率  
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")

    # Print nice results table 打印所有数据集的评估结果表格  
    print('results of'.format(args['model.name']))
    rows = []
    sum_all = 0.0
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            sum_all= sum_all + mean_acc
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    avg = sum_all / 10.0
    print(avg)

if __name__ == '__main__':
    main()

