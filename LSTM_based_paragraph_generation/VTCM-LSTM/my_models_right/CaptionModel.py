
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils



class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

   
    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

       

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
           

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): 
                for q in range(rows): 
                  
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            if t >= 1:
 
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
              
                for state_ix in range(len(new_state)):
               
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] 
                beam_seq[t, vix] = v['c'] 
                beam_seq_logprobs[t, vix] = v['r'] 
                beam_logprobs_sum[vix] = v['p'] 
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        bdash = beam_size // group_size 

        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        logprobs_table = list(init_logprobs.chunk(group_size, 0))

        args = list(args)
        args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobsf = logprobs_table[divm].data.float()
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                    logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
      
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t-divm+1)
                            done_beams_table[divm].append(final_beam)
                            beam_logprobs_sum_table[divm][vix] = -1000

                    
                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))

        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)
        return done_beams

    def beam_search_test(self, h_word, init_logprobs, temp_att, temp_theta, h_t_1, opt):

        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[
                            prev_labels]] - diversity_lambda
            return unaug_logprobsf


        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_word):


            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  
                for q in range(rows):  
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q, ix[q, c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_unaug_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_h_word = h_word.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                new_h_word[:, vix,:] = h_word[:, v['q'],:] 
                new_c_word[:, vix] = c_word[:, v['q']]
                beam_seq[t, vix] = v['c'] 
                beam_seq_logprobs[t, vix] = v['r']  
                beam_logprobs_sum[vix] = v['p'] 
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, new_h_word, candidates

        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        bdash = beam_size // group_size  


        beam_seq_table = [torch.LongTensor(self.N_max-1, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.N_max-1, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

       
        done_beams_table = [[] for _ in range(group_size)]

        logprobs_table = list(init_logprobs.chunk(group_size, 0))

        for t in range(self.N_max + group_size - 2):
            for divm in range(group_size):
                if t >= divm and t <= self.N_max + divm - 2:
                    logprobsf = logprobs_table[divm].data.float()
                    if decoding_constraint and t - divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1).cuda(), float('-inf'))
                    logprobsf[:, 2] = logprobsf[:, 2] - 1000
         
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)

                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    h_word, \
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t - divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                h_word)

                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix] == 1 or t == self.N_max + divm - 2:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t - divm + 1)
                            done_beams_table[divm].append(final_beam)
                            beam_logprobs_sum_table[divm][vix] = -1000



                    it = beam_seq_table[divm][t - divm]
                    logprobs_table[divm], h_word, h_t_1_theta = self.get_logprobs_state(it.cuda(), temp_att, h_word, temp_theta)
                    h_t_1 = h_t_1 + h_t_1_theta
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a, b: a + b, done_beams_table)
        return done_beams, h_t_1