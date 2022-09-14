import sys
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

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


def get_arch(model_name, in_c=3, n_classes=1, pretrained=True, norm='bn', nr_groups=32):

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









    # print('------- maybe should try with encoder only or decoder only if this does not work? -------')
    if norm != 'bn':
        # https://github.com/pytorch/vision/issues/2391#issuecomment-653900218
        for name, module in model.m1.encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = get_layer(model.m1.encoder, name)
                # Create new in layer
                if norm == 'in':
                    normalization = 'Instance-Norm'
                    new_norm = torch.nn.InstanceNorm2d(bn.num_features)
                elif norm == 'ln':
                    normalization = 'Layer-Norm'
                    new_norm = nn.GroupNorm(1, bn.num_features)
                elif norm == 'gn':
                    normalization = 'Group-Norm ({} groups)'.format(nr_groups)
                    new_norm = nn.GroupNorm(nr_groups, bn.num_features)
                else:
                    sys.exit('norm should be bn, in, ln, or gn, sorry')
                # Assign in
                # print("Swapping {} with {}".format(bn, inst))
                set_layer(model.m1.encoder, name, new_norm)
        for name, module in model.m2.encoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = get_layer(model.m2.encoder, name)
                # Create new in layer
                if norm == 'in':
                    normalization = 'Instance-Norm'
                    new_norm = torch.nn.InstanceNorm2d(bn.num_features)
                elif norm == 'ln':
                    normalization = 'Layer-Norm'
                    new_norm = nn.GroupNorm(1, bn.num_features)
                elif norm == 'gn':
                    normalization = 'Group-Norm ({} groups)'.format(nr_groups)
                    new_norm = nn.GroupNorm(nr_groups, bn.num_features)
                else:
                    sys.exit('norm should be bn, in, ln, or gn, sorry')
                # Assign in
                # print("Swapping {} with {}".format(bn, inst))
                set_layer(model.m2.encoder, name, new_norm)
        print('* ENCODER: Switched from Batch-Norm to {}'.format(normalization))

    if norm != 'bn':
        # https://github.com/pytorch/vision/issues/2391#issuecomment-653900218
        for name, module in model.m1.decoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = get_layer(model.m1.decoder, name)
                # Create new in layer
                if norm == 'in':
                    normalization = 'Instance-Norm'
                    new_norm = torch.nn.InstanceNorm2d(bn.num_features)
                elif norm == 'ln':
                    normalization = 'Layer-Norm'
                    new_norm = nn.GroupNorm(1, bn.num_features)
                elif norm == 'gn':
                    normalization = 'Group-Norm ({} groups)'.format(nr_groups)
                    new_norm = nn.GroupNorm(nr_groups, bn.num_features)
                else:
                    sys.exit('norm should be bn, in, ln, or gn, sorry')
                # Assign in
                # print("Swapping {} with {}".format(bn, inst))
                set_layer(model.m1.decoder, name, new_norm)
        for name, module in model.m2.decoder.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = get_layer(model.m2.decoder, name)
                # Create new in layer
                if norm == 'in':
                    normalization = 'Instance-Norm'
                    new_norm = torch.nn.InstanceNorm2d(bn.num_features)
                elif norm == 'ln':
                    normalization = 'Layer-Norm'
                    new_norm = nn.GroupNorm(1, bn.num_features)
                elif norm == 'gn':
                    normalization = 'Group-Norm ({} groups)'.format(nr_groups)
                    new_norm = nn.GroupNorm(nr_groups, bn.num_features)
                else:
                    sys.exit('norm should be bn, in, ln, or gn, sorry')
                # Assign in
                # print("Swapping {} with {}".format(bn, inst))
                set_layer(model.m2.decoder, name, new_norm)
        print('* DECODER: Switched from Batch-Norm to {}'.format(normalization))

    # # Replace conv2d by std_conv2d
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         # Get current bn layer
    #         conv = get_layer(model, name)
    #         # Create new in layer
    #         b = False if conv.bias is None else True
    #         new_conv = StdConv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
    #                              conv.stride, conv.padding, conv.dilation, conv.groups,
    #                              bias=b, padding_mode=conv.padding_mode)
    #
    #         with torch.no_grad():
    #             new_conv.weight.copy_(conv.weight)
    #             if b: new_conv.bias.copy_(conv.bias)
    #         set_layer(model, name, new_conv)
    # print(model.m1.encoder.conv1)
    return model, mean, std


