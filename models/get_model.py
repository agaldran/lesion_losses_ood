import sys
import segmentation_models_pytorch as smp
import torch
import timm


class SMP_W(torch.nn.Module):
    def __init__(self, decoder=smp.FPN, encoder_name='resnet34', encoder2_name=None, in_channels=3,
                 encoder_weights='imagenet', classes=1, mode='train'):
        super(SMP_W, self).__init__()
        if encoder2_name is None: encoder2_name = encoder_name
        self.m1 = decoder(encoder_name=encoder_name, in_channels=in_channels, encoder_weights=encoder_weights,
                          classes=classes)
        self.m2 = decoder(encoder_name=encoder2_name, in_channels=in_channels + classes,
                          encoder_weights=encoder_weights, classes=classes)
        self.n_classes = classes
        self.mode = mode

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(torch.cat([x, x1], dim=1))
        if self.mode != 'train':
            return x2
        return x1, x2


def get_arch(model_name, in_c=3, n_classes=1, pretrained=True):

    e_ws = 'imagenet' if pretrained else None

    ## FPNET ##
    # RESNET18
    if model_name == 'fpnet_resnet18_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet18', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'fpnet_resnet34_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet34', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'fpnet_resnet50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet50', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'fpnet_resnet101_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet101', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'fpnet_resnet152_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet152', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    # MOBILENET
    elif model_name == 'fpnet_mobilenet_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    # RESNEXT50
    elif model_name == 'fpnet_resnext50_W_imagenet':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='imagenet')
    elif model_name == 'fpnet_resnext50_W_ssl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'fpnet_resnext50_W_swsl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='swsl')
    # RESNEXT101
    # 32X4
    elif model_name == 'fpnet_resnext101_32x4d_W_ssl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'fpnet_resnext101_32x4d_W_swsl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='swsl')
    # 32X8
    elif model_name == 'resnext101_32x8d_W_imagenet':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x8d', in_channels=in_c, classes=n_classes, encoder_weights='imagenet')
    elif model_name == 'resnext101_32x8d_W_instagram':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x8d', in_channels=in_c, classes=n_classes, encoder_weights='instagram')
    elif model_name == 'resnext101_32x8d_W_ssl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x8d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'resnext101_32x8d_W_swsl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x8d', in_channels=in_c, classes=n_classes, encoder_weights='swsl')
    # 32X16
    elif model_name == 'resnext101_32x16d_W_instagram':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x16d', in_channels=in_c, classes=n_classes, encoder_weights='instagram')
    elif model_name == 'resnext101_32x16d_W_ssl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x16d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'resnext101_32x16d_W_swsl':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x16d', in_channels=in_c, classes=n_classes, encoder_weights='swsl')

    elif model_name == 'mobilenetV2':
        model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=1, in_chans=4)
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=n_classes, in_chans=4)
    elif model_name == 'efficientnet_b1':
        model = timm.create_model('tf_efficientnet_b1', pretrained=True, num_classes=n_classes, in_chans=4)
    elif model_name == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True, num_classes=n_classes, in_chans=4)
    elif model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=n_classes, in_chans=4)
    elif model_name == 'resnext50':
        model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=n_classes, in_chans=4)

    ########################

    else:
        sys.exit('not a valid model_name, check models.get_model.py')

    setattr(model, 'n_classes', n_classes)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


    return model, mean, std


