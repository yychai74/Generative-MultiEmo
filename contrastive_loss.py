import torch
import torch.nn as nn


# Credits https://github.com/HobbitLong/SupContrast
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None, weight=None):
        batch_size = features.shape[0]
        weight = weight.to(features.device)
        # print(labels)
        # print(features)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            # if labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.matmul(labels.T, labels)
            mask = (mask > 0).float().to(features.device)
            # print(mask)

        else:
            mask = mask.float()

        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = labels.shape[1]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.mm(contrast_feature, anchor_feature.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask).to(features.device),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        ).to(features.device)

        ## it produces 1 for the non-matching places and 0 for matching places i.e its opposite of mask
        mask = mask * logits_mask
        # ones_mask = ones_mask * logits_mask

        # compute log_prob with logsumexp
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        negative_mask = torch.eq(weight, 0) * logits_mask

        exp_mask = mask + negative_mask
        exp_logits = torch.exp(logits) * exp_mask

        ## log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 0.01)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-20)

        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss
