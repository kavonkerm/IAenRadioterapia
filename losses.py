import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Pérdida Dice orientada a maximizar la superposición volumétrica global."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        interseccion = (probs_flat * targets_flat).sum()
        dice = (2.0 * interseccion + self.smooth) / (probs_flat.pow(2).sum() + targets_flat.pow(2).sum() + self.smooth)
        return 1.0 - dice


class BinaryFocalLoss(nn.Module):
    """Focal Loss para ponderar vóxeles limítrofes complejos y mitigar el desbalance."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        modulador_focal = (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)) * ((1.0 - p_t) ** self.gamma)
        return (modulador_focal * bce).mean()


class HybridDiceFocalLoss(nn.Module):
    """Función de pérdida híbrida recomendada para radioterapia pélvica."""
    def __init__(self, peso_dice=1.0, peso_focal=1.0, alpha=0.75, gamma=2.0):
        super().__init__()
        self.peso_dice = peso_dice
        self.peso_focal = peso_focal
        self.dice_fn = DiceLoss()
        self.focal_fn = BinaryFocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits, targets):
        loss_dice = self.dice_fn(logits, targets)
        loss_focal = self.focal_fn(logits, targets)
        return self.peso_dice * loss_dice + self.peso_focal * loss_focal