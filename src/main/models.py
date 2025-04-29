import timm
import torch.nn as nn

def get_model(
    name_model: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Cria um modelo de classificação genérico via timm.

    Args:
        model_name: nome do modelo em timm (ex: 'resnet18', 'efficientnet_b0', 'vit_base_patch16_224', ...)
        num_classes: número de classes do seu problema
        pretrained: se deve carregar pesos pré-treinados
        freeze_backbone: se deve congelar todos os pesos, exceto a cabeça

    Returns:
        modelo pronto pra treinar ou fine‑tune
    """
    model = timm.create_model(
        name_model,
        pretrained=pretrained,
        num_classes=num_classes
    )

    # 2) Se quiser congelar backbone, libera só a cabeça
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(k in name.lower() for k in ['classifier', 'head', 'fc']):
                param.requires_grad = False

    return model

