import os
from typing import Iterable, Optional, TypeVar

from torchmetrics import Accuracy, MeanMetric, CatMetric

def select_metrics(metrics: dict, kw: str):
    return {key: metric for key, metric in metrics.items() if kw in key}

def forward_metrics(metrics: dict, *args, **kwargs):
    return [metric.update(*args, **kwargs) for metric in metrics.values()]

def get_metrics(metrics: dict):
    metric_results = {
        metric_name: metric.compute() for metric_name, metric in metrics.items()
    }
    return metric_results