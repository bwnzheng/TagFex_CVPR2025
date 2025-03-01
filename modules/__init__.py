from .learner.base import ContinualLearner, RehearsalLearner
from .learner.memory import HerdingIndicesLearner
from .networks import ClassIncrementalNetwork, NaiveClassIncrementalNetwork, SimpleLinear
from .metrics import Accuracy, MeanMetric, CatMetric, select_metrics, forward_metrics, get_metrics
from .optimizer import optimizer_dispatch
from .scheduler import scheduler_dispatch
from .backbones import backbone_dispatch, register_backbone
from .data.dataloader import get_loaders