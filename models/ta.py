import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.losses import prototype_loss
from models.sscl import SupCluLoss
from torchvision import transforms
import random

class projector(nn.Module):
    def __init__(self, z_dim=128, feat_dim=512, type='mlp'):
        super(projector, self).__init__()
        if type == 'mlp':
            self.proj = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, z_dim))
        else:
            self.proj = nn.Linear(feat_dim, z_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

# 调整图像张量尺寸
def resize_transform_tensor(images, size=84):
    transform_ = transforms.Compose([transforms.Resize(size)])
    images_list = []
    # 遍历每张图像并调整尺寸
    [images_list.append(transform_(img)) for img in images]
    return torch.stack(images_list).cuda() # 堆叠为张量并转移到GPU


# 生成图像局部块
def gen_patches_tensor(context_images, shot_nums, size=84, overlap=0.3, K_patch=5):
    _, _, h, w = context_images.shape # 获取图像尺寸
    way = len(shot_nums)
    ch = int(overlap * h) # 计算重叠区域高度
    cw = int(overlap * w) # 计算重叠区域宽度
    patches = [] # 存储所有块
    bbx = [] # 存储块边界坐标

    # 生成K_patch个中心区域的边界框
    for _ in range(K_patch):
        bbx.append([h // 4 - ch // 2, w // 4 - cw // 2, 3 * h // 4 + ch // 2, 3 * w // 4 + cw // 2])

    max_x0y0 = h // 2 + ch - size # 最大起始坐标（防止越界）
    for x in range(K_patch):
        # 从中心区域随机裁剪子块
        patch_x = context_images[:,:, bbx[x][0]:bbx[x][2], bbx[x][1]:bbx[x][3]]
        start_x = random.randint(0, max_x0y0 - 1) # 随机x起始位置
        start_y = random.randint(0, max_x0y0 - 1) # 随机y起始位置
        patch_xx = patch_x[:,:, start_x:start_x+size, start_y:start_y+size]
        patches.append(patch_xx)

    # 按类别和样本组织块
    point = 0
    patches_img = []
    for w in range(way): # 遍历每个类别
        patches_class = []
        for p in range(K_patch): # 遍历每个块
            pat = patches[p][point: point+shot_nums[w]] # 提取当前类别的样本块
            patches_class.append(pat)
        point = point+shot_nums[w]
        # 将块按样本组织
        for s in range(shot_nums[w]):
            for pt in patches_class:
                patches_img.append(pt[s]) # 每个样本的所有块连续存储
    images_gather = torch.stack(patches_img, dim=0) # 堆叠成张量

    return images_gather


def ta(context_images, context_labels, model, model_name="MOCO", max_iter=40, lr_finetune=0.001, distance='cos',
        is_baseline = False, is_weight_patch=False, is_weight_sample=False, K_patch=5, dataset=""):
    
    """通过迭代优化调整样本和局部块的权重"""
    model.eval()  # 固定模型参数
    lr = lr_finetune
    
    # 初始化投影网络（根据第一张图像的特征维度）
    if model_name == 'CLIP':
        feat = model.encode_image(context_images[0].unsqueeze(0)) # CLIP图像编码
    else:
        feat = model(context_images[0].unsqueeze(0)) # 普通模型前向
    proj = projector(feat_dim=feat.shape[1]).cuda() # 动态创建投影网络
    
    # 定义优化参数（模型参数+投影网络参数）
    params = []
    backbone_params = [v for k, v in model.named_parameters()] # 模型主干参数
    params.append({'params': backbone_params})
    proj_params = [v for k, v in proj.named_parameters()] # 投影网络参数
    params.append({'params': proj_params})

    optimizer = torch.optim.Adadelta(params, lr=lr) # 使用Adadelta优化器
    # 初始化损失函数和任务参数
    criterion_clu = SupCluLoss(temperature=0.07) # 监督对比聚类损失
    shot_nums = [] # 每个类别的样本数
    shot_nums_sum = 0 
    n_way = len(context_labels.unique()) # 标签去重并排序
    labels_all = [] # 扩展后的标签（每个样本生成K_patch个标签）
    # 计算每个类别的样本数并生成扩展标签
    for i in range(n_way):
        ith_way_shotnums = context_images[(context_labels == i).nonzero(), :].shape[0]
        shot_nums.append(ith_way_shotnums)
        shot_nums_sum = shot_nums_sum + ith_way_shotnums
        label_ci = [i] * shot_nums[i] * K_patch # 每个样本生成K_patch个相同标签
        labels_all = labels_all + label_ci
    label_clu_way = torch.LongTensor(list(np.reshape(labels_all, (1, -1)).squeeze())).cuda() # 转换为GPU张量

    # 初始化权重参数
    balance = 0.1
    START_WEIGHT = 10 # 开始计算权重的迭代次数阈值
    lamb = 0.7
    size_list = [84,128] # 随机选择的块尺寸
    sample_weight = None # 样本权重初始化

    if is_baseline:
        START_WEIGHT = 10086 # 设置极大值跳过权重计算
    
    # 开始优化迭代
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        """ For images 图像级特征提取 """
        if model_name == 'CLIP':
            context_features = model.encode_image(context_images)
        else:
            context_features = model(context_images)
        # 处理4D特征（如CNN输出）
        if len(context_features.shape) == 4:
            avg_pool = nn.AvgPool2d(context_features.shape[-2:])
            context_features = avg_pool(context_features).squeeze(-1).squeeze(-1)

        """ For patches """
        if i >= START_WEIGHT:
            # 生成并处理图像块
            size = random.choice(size_list) # 随机选择块尺寸
            images_gather = gen_patches_tensor(context_images, shot_nums, size=size, overlap=0.3, K_patch=K_patch)
            # 调整块尺寸适配特定模型
            if model_name in ['CLIP','DEIT','SWIN']:
                images_gather = resize_transform_tensor(images_gather, size=224)
            if model_name == 'CLIP':
                q_emb = model.encode_image(images_gather.cuda()).float()
            else:
                q_emb = model(images_gather.cuda()).float()
            # 处理4D块特征
            if len(q_emb.shape) == 4:
                avg_pool = nn.AvgPool2d(q_emb.shape[-2:])
                q_emb = avg_pool(q_emb).squeeze(-1).squeeze(-1)

            # 投影并归一化特征
            q = proj(q_emb)
            q_norm = nn.functional.normalize(q, dim=1)
            # 计算loss和patch权重
            loss_1, patch_weight = criterion_clu(q_norm, label_clu_way, shot_nums=shot_nums, is_weight_patch=is_weight_patch, q_emb=q_emb, K_patch=K_patch)

            # compute sample weight of the current iter
            if is_weight_sample:
                patch_w = patch_weight.squeeze(-1) #降维
                sample_weight_i = []
                # 对每个样本计算其所有块权重的均值
                for s in range(shot_nums_sum):
                    s_w = patch_w[s*K_patch: (s+1)*K_patch].mean()
                    sample_weight_i.append(s_w)
                sample_weight_i = torch.stack(sample_weight_i)
                # 更新样本权重
                if i == START_WEIGHT:
                    sample_weight = sample_weight_i # 首次初始化
                else:
                    sample_weight = lamb * sample_weight + (1-lamb) * sample_weight_i
                context_features = sample_weight.unsqueeze(-1) * context_features # 样本特征*权重

            loss_2, stat, _ = prototype_loss(context_features, context_labels, q_emb, label_clu_way, patch_weight=patch_weight, distance=distance)
            loss = balance * loss_1 + loss_2 # 总损失 = 聚类损失 * 系数 + 原型损失

        else:
            # 初始阶段仅计算基础原型损失
            loss, stat, _ = prototype_loss(context_features, context_labels, context_features, context_labels, patch_weight=None, distance=distance)

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

    return sample_weight # 返回最终样本权重





















