import torch
import torch.nn as nn

from modules import ClassIncrementalNetwork, SimpleLinear, backbone_dispatch

class DERNet(ClassIncrementalNetwork):
    def __init__(self, backbone_configs, network_configs, device) -> None:
        super().__init__(network_configs, device)
        self.backbone_configs = backbone_configs
        self.backbones = nn.ModuleList()
        self.classifier = None
        self.aux_classifier = None
    
    @property
    def feature_dim(self):
        return sum(net.out_dim for net in self.backbones)
    
    @property
    def output_dim(self):
        return self.classifier.out_features if self.classifier is not None else 0

    def forward(self, x):
        features = [net(x)['features'] for net in self.backbones]
        concat_feature = torch.cat(features, dim=-1)

        logits = self.classifier(concat_feature)
        out = {
            'features': features,
            'concat_feature': concat_feature,
            'logits': logits
        }

        if self.aux_classifier is not None:
            aux_logits = self.aux_classifier(features[-1])
            out.update(aux_logits=aux_logits)

        return out

    def update_network(self, num_new_classes) -> None:
        new_backbone = backbone_dispatch(self.backbone_configs)
        new_backbone.fc = None
        new_backbone.to(self.device)
        self.backbones.append(new_backbone)
        if len(self.backbones) > 1:
            # init from last backbone
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())
        
        new_dim = new_backbone.out_dim
        classifier = SimpleLinear(self.feature_dim, self.output_dim + num_new_classes, device=self.device)
        if self.classifier is not None:
            classifier.weight.data[:self.output_dim, :-new_dim] = self.classifier.weight.data
            # classifier.weight.data[:self.output_dim, -new_dim:] = 0. # a little better
            classifier.bias.data[:self.output_dim] = self.classifier.bias.data
        self.classifier = classifier

        if len(self.backbones) > 1:
            self.aux_classifier = SimpleLinear(new_dim, num_new_classes + 1, device=self.device)
    
    def weight_align(self, num_new_classes):
        new = self.classifier.weight.data[-num_new_classes:].norm(dim=-1).mean()
        old = self.classifier.weight.data[:-num_new_classes].norm(dim=-1).mean()
        self.classifier.weight.data[-num_new_classes:] *= old / new
    
    def train(self, mode: bool=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for m in self.children():
            m.train(mode)
        for m in self.backbones[:-1]:
            m.eval()
        return self

    def eval(self):
        return self.train(False)

    def freeze_old_backbones(self):
        for p in self.backbones[:-1].parameters():
            p.requires_grad_(False)
        
        for net in self.backbones[:-1]:
            net.eval()