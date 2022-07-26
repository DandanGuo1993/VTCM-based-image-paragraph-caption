import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from', type=str, default = None) 

    parser.add_argument('--checkpoint_path', type=str, default='log_xe/')

    parser.add_argument('--best_model', type=str, default=None)

    parser.add_argument('--best_model_number', type=str, default=None)

    parser.add_argument('--save_checkpoint_every', type=int, default=1500)

    parser.add_argument('--print_freq', type=int, default=1500)

    parser.add_argument('--block_trigrams', type=int, default=1,
                    help='flag to block trigrams (0=F, 1=T), default 0')

    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')
   
    parser.add_argument('--input_json', type=str, default='./data/paratalk_new.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_label_h5', type=str, default='./data/paratalk_label_new_label.h5',
                    help='path to the h5file containing the preprocessed dataset index')
#load 4096-dimensional visual features
    parser.add_argument('--input_fc_dir', type=str, default='./data/densecaption_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='./data/densecaption_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='./data/densecaption_box',
                    help='path to the directory containing the boxes of att feats')
#load 2048-dimensional visual features
    # parser.add_argument('--input_fc_dir', type=str, default='./data/parabu_fc',
    #                 help='path to the directory containing the preprocessed fc feats')   
    # parser.add_argument('--input_att_dir', type=str, default='./data/parabu_att',
    #                 help='path to the directory containing the preprocessed att feats')
    # parser.add_argument('--input_box_dir', type=str, default='./data/cocotalsk_box',
    #                 help='path to the directory containing the boxes of att feats')




    parser.add_argument('--cached_tokens', type=str, default='para_train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings


    parser.add_argument('--alpha', type=float, default=2.0,
                    help='repetition-blocking hyperparameter, default 0.0')

    parser.add_argument('--image_hidden_size', type=int, default=1024, # 512
                    help='size of the fc_feature_embedding in number of hidden nodes')
    parser.add_argument('--para_hidden_size', type=int, default=100,
                        help='size of the gbn hidden units')

    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--model_layer', type=int, default=3,
                    help='size of the model')

    parser.add_argument('--K', type=int, default=[80,50,30],
                    help='size of the model')
    parser.add_argument('--topic_layers', type=int, default=3,
                    help='number of layers in the GBN')

    parser.add_argument('--sent_rnn_hidden_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer, sent RNN')
    parser.add_argument('--word_rnn_hidden_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer, word RNN')
    parser.add_argument('--word_num_layers', type=int, default=2,
                    help='size of the rnn in number of hidden nodes in each layer, word RNN')

    parser.add_argument('--coher_hidden_size', type=int, default=100,
                    help='size of the rnn in number of hidden nodes in each layer, word RNN')
    parser.add_argument('--real_min', type=float, default=2.2e-10,
                    help='real_min')

    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--S_max', type=int, default=6,
                    help='number of layers in the RNN')
    parser.add_argument('--N_max', type=int, default=30,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=4096,  help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=4096, help='2048 for resnet, 4096 for vgg')

    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--eta0', type=int, default=0.1,
                    help='number of epochs')

    parser.add_argument('--max_epochs', type=int, default=30,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default= -1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=1,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')

    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--scheduled_sampling_start', type=int, default=0,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')

    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--lambda_gbn', type=int, default=0.1,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # Misc
    parser.add_argument('--id', type=str, default='xe',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')


    # SCST Reward

    parser.add_argument('--cider_reward_weight', type=float, default=0,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args
