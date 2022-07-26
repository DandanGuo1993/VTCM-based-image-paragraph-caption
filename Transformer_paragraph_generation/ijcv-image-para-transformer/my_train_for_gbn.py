from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse, os, pickle
import time
from six.moves import cPickle
import evaluation
import opts_GBN
# import my_models_right as my_models
from my_dataloader_new import *
import my_eval_utils_gbn
import misc.my_utils as utils
from misc.my_rewards import init_scorer, get_self_critical_reward
import PGBN_sampler
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import logging
logging.basicConfig(level=logging.INFO)

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def vocab_transform(input_tensor,old_vocab_dict,tokenizer):
    #old_vocab_dict id2word
    #new_vocab_dict word2id

    batch_size,seq_len = input_tensor.shape
    res_tensor = torch.zeros_like(input_tensor)
    for i in range(batch_size):
        for j in range(seq_len):
            #print(1)
            token = old_vocab_dict[str(input_tensor[i][j].cpu().numpy())]
            if token=='<bos>':
                token = tokenizer.bos_token
            if token=='<eos>':
                token = tokenizer.eos_token
            if token=='<unk>':
                token = tokenizer.unk_token
            if token=='<pad>':
                token = tokenizer.unk_token
            ids = tokenizer.encode(token)
            res_tensor[i][j] = int(ids[0])
    return res_tensor.cuda()

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def initialize_Phi(opt,V_tm):
    Phi = [0] * opt.topic_layers
    NDot = [0] * opt.topic_layers

    for l in range(opt.topic_layers):
        if l == 0:
            Phi[l] = np.random.rand(V_tm, opt.K[l])
        else:
            Phi[l] = np.random.rand(opt.K[l - 1], opt.K[l])

        Phi[l] = Phi[l] / np.sum(Phi[l], axis=0)
    return Phi, NDot

def updatePhi(opt, Setting, miniBatch, Phi, Theta, MBratio, MBObserved,NDot):
    Xt = np.array(np.transpose(miniBatch), order='C').astype('float64')

    ForgetRate = np.power((Setting['tao0FR'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])),-Setting['kappa0FR'])
    epsit = np.power((Setting['tao0'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])), -Setting['kappa0'])
    Eta = [0.1]* len(Phi)
    Xt_to_t1 = [0] * len(Phi)
    WSZS = [0] * len(Phi)
    EWSZS = [0] * len(Phi)
    for t in range(len(Phi)):  # t = 0:T-1
        Phi[t] = np.array(Phi[t], order='C').astype('float64')
        Theta[t] = np.array(Theta[t], order='C').astype('float64')
        if t == 0:
            Xt_to_t1[t], WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt, Phi[t], Theta[t])
        else:
            Xt_to_t1[t], WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(Xt_to_t1[t - 1], Phi[t], Theta[t])

        EWSZS[t] = MBratio * WSZS[t]  # Batch_Num * WSZS[t]

        if (MBObserved == 0):
            NDot[t] = EWSZS[t].sum(0)
        else:
            NDot[t] = (1 - ForgetRate[MBObserved]) * NDot[t] + ForgetRate[MBObserved] * EWSZS[t].sum(0)  # 1*K

        tmp = EWSZS[t] + Eta[t]  # V*K
        tmp = (1 / np.maximum(NDot[t], opt.real_min)) * (tmp - tmp.sum(0) * Phi[t])  # V*K
        tmp1 = (2 / np.maximum(NDot[t],opt.real_min)) * Phi[t]
        tmp = Phi[t] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi[t].shape[0],Phi[t].shape[1])
        Phi[t] = PGBN_sampler.ProjSimplexSpecial(tmp, Phi[t], 0)

    return Phi,NDot

def train(opt,Setting):
    # Load data
    loader = DataLoader(opt)


    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    ix2word = loader.ix_to_word

    Phi, NDot = initialize_Phi(opt, loader.TM_vocab_size)

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_xe' + '.pkl'),'rb+') as f:
        # with open(os.path.join(opt.start_from, 'model-best-i' + str(opt.best_model_number) + '-infos.pkl'), 'rb+') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'),'rb+') as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    batch_id = infos.get('batch_id', 0)
    Phi_Pi_result_history= infos.get('Phi_Pi_result_history', {})
    if opt.start_from is not None:
        opt.best_model_number = iteration
        Phi= Phi_Pi_result_history[opt.best_model_number]['Phi']
        NDot = Phi_Pi_result_history[opt.best_model_number]['NDot']
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    LMmodel = GPT2LMHeadModel.from_pretrained('gpt2')
    LMmodel.to('cuda')
    LMmodel.train()
    # Loss function
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    # Optimizer and learning rate adjustment flag
    optimizer = torch.optim.Adam(LMmodel.parameters(), lr=1, betas=(0.9, 0.98))
    update_lr_flag = True

    # Load the optimizer
    number_best = None
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        LMmodel.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best-i' + str(number_best) + '-score0.0952.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    doc_num_batches = 20000/opt.batch_size


    # Training loop
    while True:
        MBObserved = int(epoch * doc_num_batches + batch_id)

        # Update learning rate once per epoch
        if update_lr_flag:

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                LMmodel.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False

        # Load data from train split (0)
        start = time.time()
        data = loader.get_batch('train')

        batch_id +=1
        data_time  =  time.time() - start
        start  =  time.time()

        # Unpack data
        torch.cuda.synchronize()
        tmp = [data['fc_feats'], data['att_feats'], data['att_masks'], data['seq_masks'],
               data['sequence_labels'],data['para_bow_batch'],data['stop_sign_batch']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, att_masks, seq_masks, sequence_labels,para_bow_batch,stop_sign_batch = tmp

        # Forward pass and loss
        optimizer.zero_grad()
        if not sc_flag:
            para_labels = sequence_labels.view(sequence_labels.size(0),-1)
            para_labels_new = vocab_transform(para_labels,ix2word,tokenizer)

            LB, Theta, outputs = LMmodel(input_ids=None, inputs_embeds=att_feats, sentence_ids = para_labels_new, labels = para_labels_new.long(), bow = para_bow_batch, phi = Phi)
            loss_all = outputs[0] + 0.01*(-LB)


        #     output,LB,Stop_sign,Theta,Theta_sentences= dp_model(fc_feats, att_feats, att_masks, seq_masks, sequence_labels, para_bow_batch, stop_sign_batch,Phi)
        #     loss_all,loss_dpgds,loss_stop_sign,loss_word_RNN = crit(output,LB,Stop_sign,sequence_labels,seq_masks,opt.lambda_gbn)
        # else:
        #     gen_result, sample_logprobs, Theta = dp_model(fc_feats, att_feats, att_masks, sc_flag=True, opt={'sample_max': 0}, mode='sample')
        #     reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
        #     loss_all = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(),opt.lambda_gbn)

        # Backward pass
        loss_all.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss_all.item()
        torch.cuda.synchronize()

        MBratio = doc_num_batches
        Phi, NDot = updatePhi(opt, Setting, para_bow_batch.cpu().detach().numpy(), Phi, [np.transpose(Theta[i].cpu().detach().numpy().astype('float64')) for i in range(opt.topic_layers)], MBratio, MBObserved,NDot)
        # Print
        total_time = time.time() - start
        if iteration % opt.print_freq == 0:
            print('Read data:', time.time() - start)
            print("iter {} (epoch {}), loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                  .format(iteration, epoch,loss_all , data_time, total_time))
                    #           .format(iteration, epoch, np.mean(reward[:, 0]), data_time, total_time))
            # if not sc_flag:
            #     print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
            #           .format(iteration, epoch, train_loss, data_time, total_time))
            #     print('loss_dpgds:',loss_dpgds.item(),'loss_stop_sign:',loss_stop_sign.item(),'loss_word_RNN:',loss_word_RNN.item())
            # else:
            #     print("iter {} (epoch {}), avg_reward = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
            #           .format(iteration, epoch, np.mean(reward[:, 0]), data_time, total_time))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
        #
        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', LMmodel.ss_prob, iteration)
            # if not sc_flag:
            #     add_summary_value(tb_summary_writer, 'loss_dpgds:',loss_dpgds.item(), iteration)
            #     add_summary_value(tb_summary_writer, 'loss_stop_sign:',loss_stop_sign.item(), iteration)
            #     add_summary_value(tb_summary_writer, 'loss_word_RNN:', loss_word_RNN.item(), iteration)
            #
            # if sc_flag:
            #     add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)
            # loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = LMmodel.ss_prob
        #
        # Validate and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # Evaluate model
            start = time.time()
            # eval_kwargs = {'split': 'test', 'dataset': opt.input_json}
            # eval_kwargs.update(vars(opt))
            LMmodel.eval()
            gts_all = []
            gen_all = []
            test_iter = 0
            while True:
                test_iter += 1
                print(test_iter)
                test_data = loader.get_batch('test')
                tmp = [test_data['fc_feats'], test_data['att_feats'], test_data['att_masks'], test_data['seq_masks'],
                       test_data['sequence_labels'], test_data['para_bow_batch'], test_data['stop_sign_batch']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, att_masks, seq_masks, sequence_labels, para_bow_batch, stop_sign_batch = tmp
                para_labels = sequence_labels.view(sequence_labels.size(0), -1)
                para_labels_new = vocab_transform(para_labels, ix2word, tokenizer)
                with torch.no_grad():
                    #testing:
                    past=None
                    for i in range(180):
                        LB, Theta, outputs = LMmodel(input_ids=None, inputs_embeds=att_feats, sentence_ids=past, labels=None, bow = para_bow_batch, phi = Phi)
                        if past==None:
                            past = torch.argmax(outputs[0][:, -1, :], -1, keepdim=True)
                        else:
                            past = torch.cat([past,torch.argmax(outputs[0][:,-1,:], -1, keepdim=True)],-1)
                        #print(past.shape)

                    gen = tokenizer.batch_decode_gt(past, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    gts = tokenizer.batch_decode_gt(para_labels_new,skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    gts_all += gts
                    gen_all += gen
                    gts = evaluation.PTBTokenizer.tokenize(gts_all)
                    gen = evaluation.PTBTokenizer.tokenize(gen_all)
                    scores, _ = evaluation.compute_scores(gts, gen)
                if scores['BLEU'][0]<0.2:
                    if test_iter == 10:
                        break
                else:
                    if test_data['bounds']['wrapped']:
                        break

            gts = evaluation.PTBTokenizer.tokenize(gts_all)
            gen = evaluation.PTBTokenizer.tokenize(gen_all)
            scores, _ = evaluation.compute_scores(gts, gen)
            print(epoch,batch_id)
            print(scores)
            print('gts:',gts[0])
            print('generate:',gen[0])
            checkpoint_path = os.path.join(opt.checkpoint_path, f'LMmodel_{epoch}_{batch_id}.pth')
            torch.save(LMmodel.state_dict(), checkpoint_path)
            pickle.dump([gen, gts], open('generation_%d.pkl' % iteration, 'wb'))


            
            # val_loss, predictions, lang_stats = my_eval_utils_gbn.eval_split(dp_model, crit, loader, Phi,eval_kwargs,iteration)

            #  validation result into summary
            # add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            # if lang_stats is not None:
            #     for k, v in lang_stats.items():
            #         add_summary_value(tb_summary_writer, k, v, iteration)
            # val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
            # Phi_Pi_result_history[iteration] = {'Phi': Phi , 'NDot': NDot}
            #
            # # Our metric is CIDEr if available, otherwise validation loss
            # if opt.language_eval == 1:
            #     current_score = lang_stats['Bleu_4']
            # else:
            #     current_score = - val_loss
            current_score = 0

            # Save model in checkpoint path
            best_flag = False
            # if best_val_score is None or current_score > best_val_score:
            #     best_val_score = current_score
            #     best_flag = True
            # checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            # torch.save(model.state_dict(), checkpoint_path)
            # print("model saved to {}".format(checkpoint_path))
            # optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            # torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            #infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            infos['Phi_Pi_result_history'] = Phi_Pi_result_history
            #histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history


            with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(
                    os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            # Save model to unique file if new best model
            if best_flag:
                model_fname = 'model-best-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                infos_fname = 'model-best-i{:05d}-infos.pkl'.format(iteration)
                checkpoint_path = os.path.join(opt.checkpoint_path, model_fname)
                torch.save(LMmodel.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, infos_fname), 'wb') as f:
                    cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts_GBN.parse_opt()

#  online parameter
Setting={}
Setting['Iterall'] = opt.max_epochs * 20000
Setting['tao0FR'] = 0
Setting['kappa0FR'] = 0.9
Setting['tao0'] = 20
Setting['kappa0'] = 0.7
Setting['epsi0'] = 1
Setting['FurCollapse'] = 1  # 1 or 0
Setting['flag'] = 0


train(opt,Setting)
print("OK!")
