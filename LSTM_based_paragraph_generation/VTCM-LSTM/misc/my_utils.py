from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq,p_continue_all=None):
    M, J, N = seq.size()
    out = []
    for m in range(M):
        txt = ''
        # mj=p_continue_all[m].index(1)+1
        for j in range(J):
            for n in range(N):
                ix = seq[m,j,n]
                if ix != 1 and   ix != 0:
                    if n >= 1:
                        txt = txt + ' '
                    txt = txt + ' '
                    txt = txt + ix_to_word[str(ix.item())]
                else:
                    break
        out.append(txt)
    return out



def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward,LB,loss_stop_sign):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = to_contiguous(((seq != 0) & (seq != 2)).float()).view(-1)  # seq>2
        output = - input * reward * mask
        loss_self_cri = torch.sum(output) / seq.size(0)

        LB_all = torch.mean(LB)
        loss_stop_sign_all = torch.mean(loss_stop_sign)
        loss_all = -0.01*LB_all -5*loss_stop_sign_all  + loss_self_cri

        return loss_all,0.01*LB_all,5*loss_stop_sign_all,loss_self_cri

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, outputs_reshape,LB,loss_stop_sign, sequence_labels, seq_masks,lambda_gbn=0.01):
        # truncate to the same size
        # target = target[:, :input.size(1)]
        # mask =  mask[:, :input.size(1)]
        # output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        #
        # # output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # output = torch.sum(output) / torch.sum(mask)


        target = sequence_labels[:, :, 1:].contiguous().view(sequence_labels.size(0), -1)
        mask_seq = seq_masks[:, :, 1:].contiguous().view(seq_masks.size(0), -1)
        loss_word_rnn = -outputs_reshape.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask_seq
        # loss_word_rnn = torch.sum(loss_word_rnn) / torch.sum(mask_seq)
        loss_word_rnn = torch.sum(loss_word_rnn) / seq_masks.size(0)

        LB_all = torch.mean(LB)
        loss_stop_sign_all = torch.mean(loss_stop_sign)

        loss_all = -lambda_gbn*LB_all -5*loss_stop_sign_all  + loss_word_rnn

        return loss_all,lambda_gbn*LB_all,5*loss_stop_sign_all,loss_word_rnn

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    