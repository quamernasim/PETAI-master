import torchvision.models as models
from core.models.DeConvNet import *
from core.models.SeismicNet import SeismicNet


def get_model(name, pretrained, n_classes):
    model = _get_model_instance(name)

    if name == 'patch_deconvnet':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    else:
        model = model(n_classes=n_classes)

    return model


def _get_model_instance(name):
    try:
        return {
            'DeConvNet': DeConvNet,
            'DeConvNetSkip': DeConvNetSkip,
            'SeismicNet': SeismicNet,
        }[name]
    except:
        print('Model {name} not available')
