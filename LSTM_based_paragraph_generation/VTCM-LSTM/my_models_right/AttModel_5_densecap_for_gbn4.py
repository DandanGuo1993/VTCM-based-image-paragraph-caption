from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.my_utils as utils
import gensim.models as g
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

import pdb
import json


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)






class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.word_rnn_hidden_size = opt.word_rnn_hidden_size
        self.sent_rnn_hidden_size = opt.sent_rnn_hidden_size

        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.image_hidden_size = opt.image_hidden_size
        self.K = opt.K
        self.S_max = opt.S_max
        self.N_max =opt.N_max
        self.real_min = opt.real_min
        self.para_hidden_size = opt.para_hidden_size
        self.coher_hidden_size = opt.coher_hidden_size

        self.h2att = nn.Linear(self.sent_rnn_hidden_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.theta_att = nn.Linear(np.sum(opt.K), self.att_hid_size)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  
        self.sent_order_embed = nn.Sequential(nn.Embedding(6, self.word_rnn_hidden_size))
       
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed_theta1 = nn.Sequential(nn.Linear(self.fc_feat_size, self.para_hidden_size),nn.Tanh())
        self.fc_embed_theta2 = nn.Sequential(nn.Linear(self.para_hidden_size, self.para_hidden_size),nn.Tanh())
        self.fc_embed_theta3 = nn.Sequential(nn.Linear(self.para_hidden_size, self.para_hidden_size),nn.Tanh())

        self.weibu_k_1 = nn.Linear(self.para_hidden_size, 1)
        self.weibu_l_1 = nn.Linear(self.para_hidden_size, self.K[0])
        self.weibu_k_2 = nn.Linear(self.para_hidden_size, 1)
        self.weibu_l_2 = nn.Linear(self.para_hidden_size, self.K[1])
        self.weibu_k_3 = nn.Linear(self.para_hidden_size, 1)
        self.weibu_l_3 = nn.Linear(self.para_hidden_size, self.K[2])

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.image_hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.image_hidden_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.image_hidden_size),) if self.use_bn == 2 else ())))
        self.ctx2att = nn.Linear(self.image_hidden_size, self.att_hid_size)
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.word_rnn_hidden_size, self.vocab_size )
        else:
            self.logit = [[nn.Linear(self.word_rnn_hidden_size, self.word_rnn_hidden_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit = nn.Sequential( *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.word_rnn_hidden_size, self.vocab_size )]))

        self.sentRNN_logistic_mu = nn.Linear(self.sent_rnn_hidden_size, 1)


        self.sent_lstm = nn.LSTMCell(self.image_hidden_size, self.sent_rnn_hidden_size)  # we, fc, h^2_t-1
        self.word_lstm = nn.LSTMCell(self.word_rnn_hidden_size + self.image_hidden_size, self.word_rnn_hidden_size)
        self.att_lstm  = nn.LSTMCell(self.input_encoding_size + self.image_hidden_size+self.word_rnn_hidden_size, self.word_rnn_hidden_size) # we, fc, h^2_t-1


        self.Uz = nn.Linear(np.sum(self.K), self.word_rnn_hidden_size)
        self.Wz = nn.Linear(self.word_rnn_hidden_size, self.word_rnn_hidden_size)
        self.Ur = nn.Linear(np.sum(self.K), self.word_rnn_hidden_size)
        self.Wr = nn.Linear(self.word_rnn_hidden_size, self.word_rnn_hidden_size)
        self.Uh = nn.Linear(np.sum(self.K), self.word_rnn_hidden_size)
        self.Wh = nn.Linear(self.word_rnn_hidden_size, self.word_rnn_hidden_size)

        self.fc_1_coher = nn.Linear(self.word_rnn_hidden_size, self.coher_hidden_size)  # First Layer
        self.fc_2_coher = nn.Linear(self.coher_hidden_size, self.input_encoding_size)  # Second Layer
        self.non_lin_coher = nn.SELU()

        print('Attention 5-4')
        print('save to :', opt.checkpoint_path)

    def init_word_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.word_rnn_hidden_size),weight.new_zeros(self.num_layers, bsz, self.word_rnn_hidden_size))



    def log_max(self,input_x):
        return torch.log(torch.max(input_x, torch.tensor(self.real_min).cuda()))

    def GRU_theta_hidden(self,t_t,h_t_1):
        z = torch.sigmoid(self.Uz(t_t) + self.Wz(h_t_1))
        r = torch.sigmoid(self.Ur(t_t) + self.Wr(h_t_1))
        h = torch.tanh(self.Uh(t_t) + self.Wh(h_t_1.mul(r)))
        s_t =  (1-z).mul(h) + z.mul(h_t_1)
        return s_t

    def KL_GamWei(self,Gam_shape, Gam_scale, Wei_shape, Wei_scale):  
        eulergamma = 0.5772
        Wei_shape_max = torch.max(Wei_shape, torch.tensor(self.real_min).cuda())
        KL_Part1 = eulergamma * (1 - 1 / Wei_shape_max) + self.log_max(Wei_scale) - self.log_max(Wei_shape) + 1 + Gam_shape* self.log_max( Gam_scale)
        KL_Part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max(Wei_scale) - eulergamma / Wei_shape_max)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape_max))
        return KL
    def init_sent_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.sent_rnn_hidden_size),weight.new_zeros(self.num_layers, bsz, self.sent_rnn_hidden_size))

    def clip_att(self, att_feats, att_masks):

        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):


        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

       
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats
    def reparameterization(self, Wei_k, Wei_l, layer, Batch_Size):
        eps = torch.rand(Batch_Size,self.K[layer]).cuda() 
        theta = Wei_l * torch.pow(-self.log_max(1 - eps), 1 / Wei_k.repeat(1,self.K[layer]))
        return theta
    def compute_coherence_vector(self, h_prev_word):
        output1 = self.fc_1_coher(h_prev_word)
        output1_non_lin = self.non_lin_coher(output1)
        output2 = self.fc_2_coher(output1_non_lin)
        coherence_vector = self.non_lin_coher(output2)
        return coherence_vector

    def _forward(self, fc_feats_old, att_feats, att_masks=None, seq_masks=None, sequence_labels=None, para_bow_batch=None,
                 stop_sign_batch=None, Phi=None):

        batch_size = fc_feats_old.size(0)
        Phi1 = torch.tensor(Phi[0]).cuda().float()
        Phi2 = torch.tensor(Phi[1]).cuda().float()
        Phi3 = torch.tensor(Phi[2]).cuda().float()

        input_X = torch.transpose(para_bow_batch, 1, 0)


        fc_feats_h1 =self.fc_embed_theta1(fc_feats_old)
        fc_feats_h2 =self.fc_embed_theta2(fc_feats_h1)
        fc_feats_h3 =self.fc_embed_theta3(fc_feats_h2)

        K_1_ = F.softplus(self.weibu_k_1(fc_feats_h1))
        K_1 = torch.max(K_1_, torch.tensor(self.real_min).cuda())
        l_1 = F.softplus(self.weibu_l_1(fc_feats_h1))
        K_2_ = F.softplus(self.weibu_k_2(fc_feats_h2))
        K_2 = torch.max(K_2_, torch.tensor(self.real_min).cuda())
        l_2 = F.softplus(self.weibu_l_2(fc_feats_h2))
        K_3_ = F.softplus(self.weibu_k_3(fc_feats_h3))
        K_3 = torch.max(K_3_, torch.tensor(self.real_min).cuda())
        l_3 = F.softplus(self.weibu_l_3(fc_feats_h3))

        Theta1 = self.reparameterization(K_1, l_1, 0, batch_size)
        Theta2 = self.reparameterization(K_2, l_2, 1, batch_size)
        Theta3 = self.reparameterization(K_3, l_3, 2, batch_size)

        theta_prior = 0.01 * torch.ones(batch_size, self.K[-1]).cuda() 
        Theta3Scale_prior = torch.tensor(1.0,dtype=torch.float32).cuda()
        Theta2Shape_prior = torch.tensor(1.0,dtype=torch.float32).cuda()
        Theta1Scale_prior = torch.tensor(1.0,dtype=torch.float32).cuda()

        theta3_KL = torch.sum(self.KL_GamWei(theta_prior, Theta3Scale_prior, K_3, l_3))
        theta2_KL = torch.sum(self.KL_GamWei(torch.matmul(Theta3, torch.transpose(Phi3, 1, 0)),
                                             Theta2Shape_prior, K_2, l_2))
        theta1_KL = torch.sum(self.KL_GamWei(torch.matmul(Theta2, torch.transpose(Phi2, 1, 0)),
                                             Theta1Scale_prior, K_1, l_1))


        L1_1_t = (input_X).float() * self.log_max(torch.matmul(Phi1, torch.transpose(Theta1, 1, 0))) - torch.matmul(Phi1,torch.transpose(Theta1,1,0))  # - tf.lgamma( X_VN_t + 1)

        LB = (1 * torch.sum(L1_1_t) + 0.1 * theta1_KL + 0.01 * theta2_KL + 0.01 * theta3_KL) / batch_size


        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats_old, att_feats, att_masks)
        sent_state = self.init_sent_hidden(batch_size)
        sent_h  =  torch.squeeze(sent_state[0])
        sent_c  =  torch.squeeze(sent_state[1])
        h_att = sent_h
        c_att = sent_c

        outputs = fc_feats.new_zeros(batch_size,seq_masks.size(1), seq_masks.size(2) - 1, self.vocab_size)
        loss_stop_sign = 0
        sent_order =  fc_feats.new_zeros( batch_size, 1)



        for num_j in range(seq_masks.size(1)):

            sent_order_embed = self.sent_order_embed(sent_order.long())
            sent_h, sent_c = self.sent_lstm(fc_feats, ( sent_h , sent_c ))

            Y=stop_sign_batch[:, num_j].float().unsqueeze(1)
            xx = self.sentRNN_logistic_mu(sent_h)
            p_continue = torch.sigmoid(xx)
            loss_stop_sign += torch.sum( Y*self.log_max(p_continue) + (1-Y)*self.log_max(1-p_continue) )/batch_size
            h_word = sent_order_embed.squeeze(1)
            c_word = sent_order_embed.squeeze(1)
            sent_order +=1

            for num_w in range(seq_masks.size(2) - 1):
                it = sequence_labels[:,num_j, num_w].clone()
                xt = self.embed(it.long())

                att_lstm_input = torch.cat([h_word, fc_feats, xt], 1)
                h_att, c_att = self.att_lstm(att_lstm_input, (h_att, c_att))
                att, weight = self.attention_theta(torch.cat([Theta1,Theta2,Theta3],dim=-1), h_att, att_feats, p_att_feats, att_masks)
                word_lstm_input =  torch.cat([h_att, att], 1)
                h_word, c_word = self.word_lstm(word_lstm_input, (h_word, c_word))
                h_t_1_theta = self.GRU_theta_hidden(torch.cat([Theta1,Theta2,Theta3],dim=-1), h_word)
                word_lstm_output = F.dropout(h_t_1_theta, self.drop_prob_lm, self.training)
                logprobs = F.log_softmax(self.logit(word_lstm_output), dim=1)
                outputs[:,num_j, num_w] = logprobs

        outputs_reshape = outputs.view(outputs.size(0),outputs.size(1)*outputs.size(2),-1)
        Theta_sentences = [Theta1,Theta2,Theta3]
        return outputs_reshape,  LB, loss_stop_sign, Theta_sentences,Theta_sentences

    def get_logprobs_state(self, it, att, h_word, c_word,theta):
        xt = self.embed(it.long())
        word_lstm_input = torch.cat([xt, att, theta], 1)

        h_word, c_word = self.word_lstm(word_lstm_input, (h_word, c_word))

        output = F.dropout(h_word, self.drop_prob_lm, self.training)

        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, h_word, c_word


    def attention(self, h, att_feats, p_att_feats, att_masks=None):
     
        att_size = att_feats.numel() // att_feats.size(0) // self.image_hidden_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)  
        att_h = att_h.unsqueeze(1).expand_as(att)  
        dot = att + att_h  
        dot = torch.tanh(dot)  
        dot = dot.view(-1, self.att_hid_size)  
        dot = self.alpha_net(dot) 
        dot = dot.view(-1, att_size)
        weight = F.softmax(dot, dim=1) 
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  
        att_feats_ = att_feats.view(-1, att_size, self.image_hidden_size) 
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  
        index_big = weight.sort(1,descending=True)
        index_choose_weight = index_big[0][:,0:4]
        index_choose = index_big[1][:,0:4]
        index_choose_all=[index_choose_weight,index_choose]
        return att_res,index_choose

    def attention_theta(self, Theta, h, att_feats, p_att_feats, att_masks=None):
        att_size = att_feats.numel() // att_feats.size(0) // self.image_hidden_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)  
        att_h = att_h.unsqueeze(1).expand_as(att) 
        theta_att = self.theta_att(Theta)
        theta_att = theta_att.unsqueeze(1).expand_as(att)
        dot = theta_att + att + att_h  
        dot = torch.tanh(dot) 
        dot = dot.view(-1, self.att_hid_size)  
        dot = self.alpha_net(dot)  
        dot = dot.view(-1, att_size)  
        weight = F.softmax(dot, dim=1) 
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  
        att_feats_ = att_feats.view(-1, att_size, self.image_hidden_size) 
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) 
        index_big = weight.sort(1,descending=True)
        index_choose_weight = index_big[0][:,0:4]
        index_choose = index_big[1][:,0:4]
        index_choose_all=[index_choose_weight,index_choose]
        return att_res,index_choose

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size , 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            sent_state = self.init_sent_hidden(beam_size)
            sent_h = torch.squeeze(sent_state[0])
            sent_c = torch.squeeze(sent_state[1])



            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k + 1].expand(
                *((beam_size,) + att_masks.size()[1:])).contiguous() if att_masks is not None else None

            it = fc_feats.new_zeros([beam_size], dtype=torch.long)  
            logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks,
                                                      state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] 
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats_old, att_feats, att_masks=None, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats_old, att_feats, att_masks, opt)

        batch_size = fc_feats_old.size(0)
        sent_state = self.init_sent_hidden(batch_size)
        sent_h  =  torch.squeeze(sent_state[0])
        sent_c  =  torch.squeeze(sent_state[1])
        h_att = sent_h
        c_att = sent_c

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats_old, att_feats, att_masks)
        weight_all_sentences = fc_feats.new_zeros(batch_size, 4,self.S_max,self.N_max)

        ##   GBN  model
        fc_feats_h1 =self.fc_embed_theta1(fc_feats_old)
        fc_feats_h2 =self.fc_embed_theta2(fc_feats_h1)
        fc_feats_h3 =self.fc_embed_theta3(fc_feats_h2)

        K_1_ = F.softplus(self.weibu_k_1(fc_feats_h1))
        K_1 = torch.max(K_1_, torch.tensor(self.real_min).cuda())
        l_1 = F.softplus(self.weibu_l_1(fc_feats_h1))
        K_2_ = F.softplus(self.weibu_k_2(fc_feats_h2))
        K_2 = torch.max(K_2_, torch.tensor(self.real_min).cuda())
        l_2 = F.softplus(self.weibu_l_2(fc_feats_h2))
        K_3_ = F.softplus(self.weibu_k_3(fc_feats_h3))
        K_3 = torch.max(K_3_, torch.tensor(self.real_min).cuda())
        l_3 = F.softplus(self.weibu_l_3(fc_feats_h3))

        Theta1 = self.reparameterization(K_1, l_1, 0, batch_size)
        Theta2 = self.reparameterization(K_2, l_2, 1, batch_size)
        Theta3 = self.reparameterization(K_3, l_3, 2, batch_size)

       
        trigrams = []  
        seq = fc_feats.new_zeros((batch_size,self.S_max, self.N_max-1), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size,self.S_max, self.N_max-1)
        p_continue_all = fc_feats.new_zeros(batch_size,self.S_max)
        sent_order= fc_feats.new_zeros(batch_size,1)

        for num_j in range(self.S_max):
            sent_order_embed = self.sent_order_embed(sent_order.long())
            h_word = sent_order_embed.squeeze(1)
            c_word = sent_order_embed.squeeze(1)
            sent_order += 1
            sent_h, sent_c = self.sent_lstm(fc_feats, ( sent_h , sent_c ))

            xx = self.sentRNN_logistic_mu(sent_h)
            p_continue_all[:,num_j]  = torch.bernoulli(torch.sigmoid(xx)).view(-1)

            for t in range(self.N_max - 1):
                if t == 0:  
                    it = fc_feats.new_zeros(batch_size, dtype=torch.long)
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:

                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data) 
                    else:
                        
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it) 
                    it = it.view(-1).long()  

                if t >= 1:
                    
                    if t == 1:
                        unfinished = (it != 1)
                    else:
                        unfinished = unfinished * (it != 1)
                    if unfinished.sum() == 0:
                        break
                    it = it * unfinished.type_as(it)
                    seq[:, num_j, t - 1] = it
                    seqLogprobs[:, num_j, t - 1] = sampleLogprobs.view(-1)

                xt = self.embed(it.long())

                att_lstm_input = torch.cat([h_word, fc_feats, xt], 1)
                h_att, c_att = self.att_lstm(att_lstm_input, (h_att, c_att))
                att, weight = self.attention_theta(torch.cat([Theta1,Theta2,Theta3],dim=-1), h_att, att_feats, p_att_feats, att_masks)
                word_lstm_input =  torch.cat([h_att, att], 1)
                h_word, c_word = self.word_lstm(word_lstm_input, (h_word, c_word))
                h_t_1_theta = self.GRU_theta_hidden(torch.cat([Theta1,Theta2,Theta3],dim=-1), h_word)
                word_lstm_output = F.dropout(h_t_1_theta, self.drop_prob_lm, self.training)
                logprobs = F.log_softmax(self.logit(word_lstm_output), dim=1)
                weight_all_sentences[:, :, num_j,t] = weight
               
                if block_trigrams and t >= 3 and sample_max:
                  
                    prev_two_batch = seq[:,num_j, t - 3:t - 1]
                    for i in range(batch_size):  
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        current = seq[i,num_j,t - 1]
                        if t == 3: 
                            trigrams.append({prev_two: [current]})  
                        elif t > 3:
                            if prev_two in trigrams[i]:  
                                trigrams[i][prev_two].append(current)
                            else:  
                                trigrams[i][prev_two] = [current]
       
                    prev_two_batch = seq[:,num_j, t - 2:t]
                    mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  
                    for i in range(batch_size):
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        if prev_two in trigrams[i]:
                            for j in trigrams[i][prev_two]:
                                mask[i, j] += 1
                  
                    alpha = 2.0 
                    logprobs = logprobs + (mask * (-0.693) * alpha)  


               
        return seq, p_continue_all, seqLogprobs, [Theta1,Theta2,Theta3], weight_all_sentences




class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 1
