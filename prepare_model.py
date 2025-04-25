import torchvision.models as models
import torch
import timm
import re

vit_model_dict = {
    'vitb32': 'vit_base_patch32_224',
    'vitb16': 'vit_base_patch16_224',
    'vitl32': 'vit_large_patch32_224',
    'vitl16': 'vit_large_patch16_224',
    'vith14': 'vit_huge_patch14_224',
    'vitb32_clip': 'vit_base_patch32_clip_224',
    'vitb16_clip': 'vit_base_patch16_clip_224',
    'vitl14_clip': 'vit_large_patch14_clip_224',
    'vith14_clip': 'vit_huge_patch14_clip_224',
}

def prepare_vm(model_name, num_classes, mode='linear', lr=1e-3):
    encoder_params, clf_params = [], []
    hook_layer = None
    if 'dinov2' in model_name:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.head = torch.nn.Linear(768, num_classes)
        hook_layer = model.head
        
        for name, param in model.named_parameters():
            if 'head' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
                
    elif 'vit' in model_name:
        model = timm.create_model(vit_model_dict[model_name], pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
        hook_layer = model.head
        
        for name, param in model.named_parameters():
            if 'head' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
    elif 'resnet' in model_name:
        num_layers = re.findall(r'(\d+)', model_name)[0]
        pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
        model = models.__dict__[model_name](weights=pretrain_weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        hook_layer = model.fc
        
        for name, param in model.named_parameters():
            if 'fc' not in name:
                encoder_params.append(param)
            else:
                clf_params.append(param)
                
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    if mode == 'linear':
        for param in encoder_params:
            param.requires_grad = False
        for param in clf_params:
            param.requires_grad = True    
        optimizer = torch.optim.AdamW(clf_params, lr=lr)
        
    elif mode == 'full':
        for param in encoder_params:
            param.requires_grad = True
        for param in clf_params:
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': lr/10},
                {'params': clf_params, 'lr': lr}
            ]
        )
    else:
        raise NotImplementedError(f'{mode} is not supported')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{mode} MODE Total parameters: {total_params}, Trainable parameters: {trainable_params}')
    
    return model, optimizer, hook_layer