from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
import re

import h5py
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

def update_vocab(symbol, idxvocab, wtoi,itow):
    idxvocab.append(symbol)
    wtoi[symbol] = len(idxvocab) - 1
    itow[len(idxvocab) - 1] = symbol


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']
    counts = {}
    nsents=0
    for img in imgs:
        for sent in img['sentences']:
            nsents += 1
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))
    total_words = sum(counts.values())
    print('total words:', total_words)



    vocab = [w for w, n in counts.items() if n > count_thr]

    idxvocab=[]
    itow = {}
    wtoi = {}
    dummy_symbols = ['<bos>', '<eos>', '<pad>', '<unk>']
    for word in dummy_symbols:
        update_vocab(word, idxvocab, wtoi,itow)

    for word in vocab:
        update_vocab(word, idxvocab, wtoi,itow)



    bad_words = [w for w, n in counts.items() if n <= count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else '<unk>' for w in txt]
            img['final_captions'].append(caption)
##   My code
    Sentence_caption_counter =0

    for img in imgs:
        img['Sentence_captions'] = []

        for txt in img['sentences'][0]['tokens_sent']:

            caption = [w if counts.get(w, 0) > count_thr else '<unk>' for w in txt]
            img['Sentence_captions'].append(caption)
        Sentence_caption_counter += len(img['Sentence_captions'])

    print('average number of sentences in each paragraph:',Sentence_caption_counter/len(imgs))

    stopwords = set([item.strip().lower() for item in open(params['stopwords'])])
    freqwords = set([item[1] for item in cw[:int(float(len(idxvocab)) * 0.001)]])  
    alpha_check = re.compile("[a-zA-Z]")
    symbols = set([w for w in wtoi.keys() if ((alpha_check.search(w) == None) or w.startswith("'"))])


    ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])  
    ignore = set([wtoi[w] for w in ignore if w in wtoi])

    TM_vocab = np.delete(idxvocab, list(ignore))

    TM_vocab1=TM_vocab.tolist()


    return TM_vocab1,idxvocab,itow, wtoi,ignore


def encode_captions(imgs, params, wtoi, TM_vocab,ignore):
    
    max_length = params['max_length']
    N_max = params['N_max']
    S_max = params['S_max']


    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        img_new =img['final_captions']
        n = len(img_new)
        assert n > 0, 'error: some image has no captions'

        Li = wtoi['<pad>']+np.zeros((n, max_length), dtype='uint32') 
        
        for j, s in enumerate(img_new):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    all_img_captions_matrix = []
    all_num_distribution=[]
    for i, img in enumerate(imgs):
        img_num_sents = len(img['Sentence_captions'])
        if img_num_sents > S_max:
            img_num_sents = S_max
        img_num_distribution = np.zeros([S_max], dtype=np.int32)
        img_num_distribution[img_num_sents - 1:] = 1
        all_num_distribution.append(img_num_distribution)
        img_captions_matrix = np.ones([S_max, N_max], dtype=np.int32) * 2  # zeros([6, 50])
        for idx, img_sent in enumerate(img['Sentence_captions']):

            if idx == img_num_sents:
                break
            img_sent_new = ['<bos>']+img_sent+['<eos>']

            for idy, word in enumerate(img_sent_new):
                if idy == N_max:
                    break
                img_captions_matrix[idx, idy] = wtoi[word]
        all_img_captions_matrix.append(img_captions_matrix)

    all_img_captions_matrix_concat = np.reshape(all_img_captions_matrix,[len(imgs),-1])
    TM_train_bow = np.zeros([len(imgs),len(wtoi)])
    contex_bow = np.zeros([len(imgs),S_max,len(wtoi)])
    for doc_index in range(len(imgs)):
        for word in all_img_captions_matrix_concat[doc_index, :]:
            TM_train_bow[doc_index][word] += 1
        paragraph = all_img_captions_matrix[doc_index] #6*30
        for sent_num in range(S_max):
            sent_bow = Bow_sents(paragraph[sent_num],len(wtoi))
            contex_bow[doc_index][sent_num] = TM_train_bow[doc_index]-sent_bow
    TM_train_bow = np.delete(TM_train_bow, list(ignore), axis = 1)
    contex_bow = np.delete(contex_bow, list(ignore), axis=2)
    return all_img_captions_matrix, L  , label_start_ix, label_end_ix, label_length,TM_train_bow,contex_bow,all_num_distribution

def Bow_sents(s, V):
    s_b = np.zeros(V)
    for w in s:
        s_b[w] += 1
    return s_b









def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    imgs = [x for x in imgs if x['id'] not in [2346046, 2341671]]

    seed(123)



    TM_vocab, idxvocab, idx2word, word2idx, ignore = build_vocab(imgs, params)




    # encode captions in large arrays, ready to ship to hdf5 file
    all_img_captions_matrix,L, label_start_ix, label_end_ix, label_length,TM_train_bow,contex_bow,all_num_distribution = encode_captions(imgs, params, word2idx,TM_vocab,ignore)

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("Sentence_labels", dtype='uint32', data=all_img_captions_matrix)
    f_lb.create_dataset("TM_train_bow", dtype='uint32', data=TM_train_bow)
    f_lb.create_dataset("context_bow", dtype='uint32', data=contex_bow)
    f_lb.create_dataset("all_STOP_Sign", dtype='uint32', data=all_num_distribution)

    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['TM_vocab'] = TM_vocab  #
    out['idxvocab'] = idxvocab  #
    out['word_to_ix'] = word2idx  #
    out['ix_to_word'] = idx2word  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img['filepath'],
                                                               img['filename'])  # copy it over, might need
        if 'cocoid' in img: jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        if params['images_root'] != '':
            with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json',
                        default='./data/captions/para_karpathy_format_new.json',
                        help='input json file to process into hdf5')

    parser.add_argument('--stopwords', default='./data/stop_list_en.txt',help = 'input json file to get the bag of words')

    # output json
    parser.add_argument('--output_json', default='./data/paratalk_new.json',
                        help='output json file')
    parser.add_argument('--output_h5', default='./data/paratalk_label_new',
                        help='output h5 file')
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=180, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--N_max', default=30, type=int,
                        help='max length of a Sentence, in number of words. captions longer than this get clipped.')
    parser.add_argument('--S_max', default=6, type=int,
                        help='max length of the number of Sentences, in number of sentence. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
