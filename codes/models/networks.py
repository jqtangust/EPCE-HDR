import torch
import logging

from models.epce_model.EPCE import Curve_Estimation

logger = logging.getLogger('base')
input = torch.Tensor(1,3,256,256)

####################
# define network
####################
#### Generator
def define_G(opt,tb_logger):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'EPCE':
        netG = Curve_Estimation()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG
