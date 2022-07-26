from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import scipy.io as sio

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.my_utils as utils


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = './annotations/para_captions_test.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap


    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w', encoding= 'utf-8'))  

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w', encoding= 'utf-8') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model, crit, loader,Phi, eval_kwargs={},iteration=None):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    ID_all = []
    file = open(eval_kwargs.get('checkpoint_path', '0') + str(iteration) + '_generated.txt', 'w', encoding='utf-8')
    for i in range(4):
        data = loader.get_batch(split)
        if data.get('sequence_labels', None) is not None and verbose_loss:
            tmp = [data['fc_feats'], data['att_feats'], data['att_masks'], data['seq_masks'],
                   data['sequence_labels'], data['para_bow_batch'], data['stop_sign_batch']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]

            fc_feats, att_feats, att_masks, seq_masks, sequence_labels, para_bow_batch, stop_sign_batch = tmp

            with torch.no_grad():
                output, LB, Stop_sign, Theta,Theta = model(fc_feats, att_feats, att_masks, seq_masks, sequence_labels,
                                                     para_bow_batch, stop_sign_batch, Phi)


                loss_all, loss_dpgds, loss_stop_sign, loss_word_RNN = crit(output, LB, Stop_sign, sequence_labels,
                                                                           seq_masks)
            loss_sum = loss_sum + loss_word_RNN.item()
            loss_evals = loss_evals + 1


        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[ 'att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        with torch.no_grad():
            A = model( fc_feats,att_feats, att_masks, opt=eval_kwargs, mode='sample')
            seq=A[0].data
            P=A[1].data.cpu().numpy().tolist()
            Theta =A[3]
            weight=A[4]
            if  n==0:
                Theta_all_1 = Theta[0]
                Theta_all_2 = Theta[1]
                Theta_all_3 = Theta[2]
                weight_all = weight
                for i in range(loader.batch_size):
                    ID_all.append(data['infos'][i]['id'])
            else:
                Theta_all_1 = torch.cat([Theta_all_1,Theta[0]],0)
                Theta_all_2 = torch.cat([Theta_all_2,Theta[1]],0)
                Theta_all_3 = torch.cat([Theta_all_3,Theta[2]],0)
                weight_all = torch.cat([weight_all,weight],0)
                for i in range(loader.batch_size):
                    ID_all.append(data['infos'][i]['id'])
       
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                file.write(str(entry['image_id'])+':' + entry['caption'])
                file.write('\n')

        n = n + loader.batch_size

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
    Theta_all = [Theta_all_1,Theta_all_2,Theta_all_3]
    file.close()
   

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
