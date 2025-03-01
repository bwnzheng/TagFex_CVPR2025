import torch
import torch.nn as nn

from .metrics import MulticlassAccuracy, forward_metrics, get_metrics

@torch.no_grad()
def multiclass_accuracy_evaluate(network: nn.Module, data_loader, num_classes, device):
    metrics = {
        'acc1': MulticlassAccuracy(average=None, num_classes=num_classes, device=device),
        'acc5': MulticlassAccuracy(average=None, num_classes=num_classes, device=device, k=5)
    }

    for batch, batch_data in enumerate(data_loader):
        batch_data = tuple(data.to(device, non_blocking=True) for data in batch_data)

        samples, targets, *_ = batch_data
        logits = network(samples)['logits']
        forward_metrics(logits, targets)

    metric_results = get_metrics(metrics)
    return metric_results
