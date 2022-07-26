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
    # sys.path.append('F:/image-para/exp')
    annFile = 'F:/image-para/exp/data/annotations/para_captions_test.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w', encoding= 'utf-8'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
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

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    ID_all = []
    file = open(eval_kwargs.get('checkpoint_path', '0') + str(iteration) + '_generated.txt', 'w', encoding='utf-8')
    # while True:
    for i in range(4):
        data = loader.get_batch(split)
        if data.get('sequence_labels', None) is not None and verbose_loss:
            # forward the model to get loss
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

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[ 'att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
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
        # Print beam search
        # if beam_size > 1 and verbose_beam:
        # for i in range(loader.batch_size):
                # print('\n'.join(
                #     [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                # print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            if  data['infos'][k]['id']  in [2378082,2413687,2358602,2392549,2404995,2319286,2320053,2383810,
                                               2377445,2352308,2373846,2383927,2336456,2416313, 2353511, 2395940,
                                               2384184, 2354621,2394489,2343337,2354120,2392875, 2324610,2411664,
                                               2407689, 2374604, 2368610, 2365470, 2391501, 2353208, 2389502, 2375684,
                                               2386555, 2401657, 2408986, 2335134, 2349326, 2401939, 2387402, 2387294,
                                               2395320, 2394489, 2395459, 2354621, 2384184, 2326503, 2395940, 2345348,
                                               2343744, 2369715, 2373213,2401733, 2391570, 2395982, 2399639, 2342158,
                                               2347914]:

                A=fc_feats[k]
                B=att_feats[k]
            else:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                if eval_kwargs.get('dump_path', 0) == 1:
                    entry['file_name'] = data['infos'][k]['file_path']
                predictions.append(entry)
                if eval_kwargs.get('dump_images', 0) == 1:
                    # dump the raw image to vis/ folder
                    cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                                data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                        len(predictions)) + '.jpg'  # bit gross
                    print(cmd)
                    os.system(cmd)

                if verbose:
                    print('image %s: %s' % (entry['image_id'], entry['caption']))
                    file.write(str(entry['image_id'])+':' + entry['caption'])
                    file.write('\n')

        n = n + loader.batch_size

        # if we wrapped around the split or used up val imgs budget then bail
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
    # sio.savemat(eval_kwargs.get('checkpoint_path', '0')+'Theta'+ str(iteration)+ '.mat', {'Theta': Theta_all.data.cpu().numpy(),'Weight': weight_all.data.cpu().numpy(),'ID': np.array(ID_all)})
    # sio.savemat(eval_kwargs.get('checkpoint_path', '0')+'Phi_Pi'+ str(iteration)+ '.mat', {'Para_Phi': Phi})

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
