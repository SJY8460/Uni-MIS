import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional
from utils.matrix_utils import matrix2flat2,matrix2flat3


def multilabel2one_hot(labels, nums):
    # print(labels)
    res = [0.] * nums
    if len(labels) == 0:
        return res
    if isinstance(labels[0], list):
        for label in labels[0]:
            res[label] = 1.
        return res
    for label in labels:
        res[label] = 1.
    # print(res)
    return res


def sequence_mask(sequence_length, max_len=None):
    """
        获得序列的mask
    Args:
        sequence_length ([type]): [description]
        max_len ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def get_token_level_loss(logits, target, length):
    length = Variable(torch.LongTensor(length)).cuda()
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def get_slot_label_loss(
    criterion,
    pred_logistic,
    glod_label,
    entity_num=None,
):
    """
        Get Entity Level Slot label prediction loss
    """
    device = pred_logistic.device
    pred_out = torch.cat(
        [pred_logistic[i][:entity_num[i]] for i in range(0, len(entity_num))],
        dim=0)
    target_var = torch.cat(
        [glod_label[i][:entity_num[i]] for i in range(0, len(entity_num))],
        dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
    target_var = target_var.to(device)
    pred_out = pred_out.to(device)
    return criterion(pred_out, target_var)


def get_loss(criterion,
             pred_logistic,
             glod_label,
             label_num,
             seq_lens,
             window_size,
             to_token_level=False,
             is_token_level=False,
             multi=False,
             print_info=False):
    """
        param: criterion        用于计算损失
        param: pred_logistic    预测结果
        param: glod_label       标准答案
        param: seq_len          句子长度
        param: to_token_level   sentence-level转为token-level任务计算
        param: is_token_level   is_token_level 判断是不是token-level
        param: multi            判断是不是多分类
    """
    # Target 进行修改
    target_out = glod_label
    seq_lens = seq_lens.cpu().numpy()
    if to_token_level:
        is_token_level = True
        token_level_list = []
        for item, num in zip(glod_label, seq_lens):
            token_level_list.extend([item] * num)  # token-level
        target_out = token_level_list

    if multi:
        multi_list = [
            multilabel2one_hot(intents, label_num)
            for intents in token_level_list
        ]
        target_out = multi_list
    # print(pred_logistic)
    device = pred_logistic.device
    if to_token_level or multi:
        target_var = torch.Tensor(target_out)
    else:
        if window_size == 1:
            target_var = torch.cat(
                [glod_label[i][:seq_lens[i]] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
        elif window_size == 2:
            target_var = torch.cat(
                [glod_label[i][:seq_lens[i]] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
        elif window_size == 3:
            target_var = torch.cat(
                [glod_label[i][1:seq_lens[i]+1] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
        elif window_size == 4:
            target_var = torch.cat(
                [glod_label[i][1:seq_lens[i]+1] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
        elif window_size == 5:
            target_var = torch.cat(
                [glod_label[i][2:seq_lens[i]+2] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
    target_var = target_var.to(device)
    # Prediction部分修改
    if is_token_level:
        pred_out = torch.cat(
            [pred_logistic[i][:seq_lens[i]] for i in range(0, len(seq_lens))],
            dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
        
    # if print_info:
    #     print("target", target_var)
    #     print('pred', pred_out)
    #     exit()
    return criterion(pred_out, target_var)

def get_token_intent_loss(criterion,
             pred_logistic,
             glod_label,#[batch,seq_lens]
             label_num,
             window_size,
             seq_lens,):
    #target_list = glodlabel2onehot(glod_label,label_num)
    if window_size == 1:
        target_flat = matrix2flat3(glod_label,seq_lens)
        target_list = [glod2onehot(token_intent,label_num) for token_intent in target_flat]
        target_var = torch.Tensor(target_list)
        pred_out = torch.cat(
                [pred_logistic[i][:seq_lens[i]] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
    else:
        target_flat = matrix2flat2(glod_label,seq_lens)
        target_list = [glod2onehot(token_intent,label_num) for token_intent in target_flat]
        target_var = torch.Tensor(target_list)
        pred_out = torch.cat(
                [pred_logistic[i][:seq_lens[i]] for i in range(0, len(seq_lens))],
                dim=0)  # 避免 padding带来的错误干扰 只有token-level才有这个问题
    device = pred_logistic.device
    target_var = target_var.to(device)
    return criterion(pred_out,target_var)

def glod2onehot(glodlabel,label_num): 
    res = [0.] * label_num
    res[glodlabel] = 1.
    return res
