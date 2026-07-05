from .m3ddcnn import m3ddcnn
from .cnn3d import cnn3d
from .rssan import rssan
from .ablstm import ablstm
from .dffn import dffn
from .speformer import speformer
from .ssftt import ssftt
from .BiDA import BiDA
from .GAHT import GAHT

def get_model(model_name, dataset_name, patch_size, opts=None, ema=False):
    # example: model_name='cnn3d', dataset_name='pu'
    if model_name == 'm3ddcnn':
        model = m3ddcnn(dataset_name, patch_size)

    elif model_name == 'cnn3d':
        model = cnn3d(dataset_name, patch_size)
    
    elif model_name == 'rssan':
        model = rssan(dataset_name, patch_size)
    
    elif model_name == 'ablstm':
        model = ablstm(dataset_name, patch_size)

    elif model_name == 'dffn':
        model = dffn(dataset_name, patch_size)    
    
    elif model_name == 'speformer':
        model = speformer(dataset_name, patch_size) 

    elif model_name == 'GAHT':
        model = GAHT(dataset_name, patch_size)

    elif model_name == 'ssftt':
        model = ssftt(dataset_name, patch_size)

    elif model_name == 'BiDA':
        model = BiDA(dataset_name, opts)
    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    if ema:
        for param in model.parameters():
            param.detach_()
            
    return model


