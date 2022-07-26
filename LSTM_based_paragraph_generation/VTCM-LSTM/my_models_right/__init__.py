from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

# from .AttModel import TopDownModel
# from .AttModel import TopDownModel
# from .AttModel_1 import TopDownModel
# from .AttModel_2 import TopDownModel
from .AttModel_5_densecap_for_gbn4_3 import TopDownModel
# from .AttModel_5_beam_search import TopDownModel


def setup(opt):
    model = TopDownModel(opt)

    # Check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best-' + opt.best_model + '.pth')))
        # model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model




