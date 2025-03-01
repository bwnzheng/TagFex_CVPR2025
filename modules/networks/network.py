from abc import abstractmethod
import torch
import torch.nn as nn
from .linears import SimpleLinear

class ContinualNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def update_network(self, *args, **kwargs) -> None:
        '''
        Called at every task beginning.
        '''
        pass

class ClassIncrementalNetwork(ContinualNetwork):
    def __init__(self, configs, device) -> None:
        super().__init__()
        self.configs = configs
        self.classifier_type = configs.get('classifier_type', 'unified')

        self.device = device

    @property
    @abstractmethod
    def feature_dim(self):
        pass
    
    @property
    @abstractmethod
    def output_dim(self):
        pass
    
    @abstractmethod
    def forward_classifier(self, features: torch.Tensor):
        pass


class NaiveClassIncrementalNetwork(ClassIncrementalNetwork):
    def __init__(self, backbone, configs, device) -> None:
        super().__init__(configs, device)
        self.backbone = backbone

        if self.classifier_type == 'unified':
            self.classifier = None # FIXME: name error
        elif self.classifier_type == 'separated':
            self.classifier = nn.ModuleList()
        
        self.to(device)

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    @property
    def output_dim(self):
        if self.classifier_type == 'unified':
            return self.classifier.out_features if self.classifier is not None else 0
        elif self.classifier_type == 'separated':
            return sum(c.out_features for c in self.classifier)

    def update_network(self, num_new_classes) -> None:
        if self.classifier_type == 'unified':
            classifier = SimpleLinear(self.feature_dim, self.output_dim + num_new_classes, device=self.device)
            if self.classifier is not None:
                classifier.weight.data[:self.output_dim] = self.classifier.weight.data
                classifier.bias.data[:self.output_dim] = self.classifier.bias.data
            self.classifier = classifier
        elif self.classifier_type == 'separated':
            classifier = SimpleLinear(self.feature_dim, num_new_classes, device=self.device)
            self.classifier.append(classifier)
    
    def forward_classifier(self, features: torch.Tensor):
        if self.classifier_type == 'unified':
            return self.classifier(features)
        elif self.classifier_type == 'separated':
            return torch.cat([c(features) for c in self.classifier], dim=-1)

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        features = out['features']
        logits = self.forward_classifier(features)

        outputs = {
            'logits': logits,
            'features': features,
            'fmaps': out['fmaps'],
        }

        return outputs
    
    def freezed_copy(self):
        from copy import deepcopy
        self_copy = deepcopy(self)
        for p in self_copy.parameters():
            p.requires_grad_(False)
        return self_copy.eval()
