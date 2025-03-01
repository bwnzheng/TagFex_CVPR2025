import torch
import torch.nn as nn

from modules import ClassIncrementalNetwork, SimpleLinear, backbone_dispatch
from utils.funcs import parameter_count

class TagFexNet(ClassIncrementalNetwork):
    def __init__(self, backbone_configs, network_configs, device) -> None:
        super().__init__(network_configs, device)
        self.backbone_configs = backbone_configs
        self.ta_net = backbone_dispatch(backbone_configs)
        if hasattr(self.ta_net, 'fc'):
            self.ta_net.fc = None
        self.ta_net.to(self.device)
        
        self.ts_nets = nn.ModuleList()
        self.classifier = None
        self.ts_attn = None
        self.trans_classifier = None
        self.aux_classifier = None

        # contrastive learning projector
        self.projector = nn.Sequential(
            SimpleLinear(self.ta_feature_dim, self.configs["proj_hidden_dim"]),
            nn.ReLU(True),
            SimpleLinear(self.configs["proj_hidden_dim"], self.configs["proj_output_dim"]),
        ).to(self.device)
        self.predictor = None
    
    @property
    def feature_dim(self):
        return self.ts_feature_dim
    
    @property
    def ts_feature_dim(self):
        if len(self.ts_nets) == 0:
            return 0
        return sum(net.out_dim for net in self.ts_nets)
    
    @property
    def ta_feature_dim(self):
        return self.ta_net.out_dim
    
    @property
    def output_dim(self):
        return self.classifier.out_features if self.classifier is not None else 0

    def forward(self, x):
        ts_outs = []
        if hasattr(self, 'ts_fmaps'):
            for net in self.ts_nets:
                out = net(x)
                self.ts_fmaps = out['fmaps']
                ts_outs.append(out)
        else:
            ts_outs = [net(x) for net in self.ts_nets]
        
        ts_features = [out['features'] for out in ts_outs]

        logits = self.classifier(torch.cat(ts_features, dim=-1))

        outputs = {
            'logits': logits,
            'ts_features': ts_features
        }

        if self.training:
            ta_fmap = self.ta_net(x)['fmaps'][-1] # (bs, C, H, W)
            ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1) # (bs, H*W, C) -mean-> (bs, C)

            embedding = self.projector(ta_feature)

            outputs.update({
                    'ta_feature': ta_feature,
                    'embedding': embedding,
            })

            if self.trans_classifier is not None:
                ts_feature = ts_outs[-1]["fmaps"][-1].flatten(2).permute(0, 2, 1)
                ta_features = ta_fmap.flatten(2).permute(0, 2, 1)
                merged_feature = self.ts_attn(ta_features.detach(), ts_feature).mean(1)
                trans_logits = self.trans_classifier(merged_feature)
                outputs.update(trans_logits=trans_logits)

            if self.aux_classifier is not None:
                aux_logits = self.aux_classifier(ts_features[-1])
                outputs.update(aux_logits=aux_logits)

            if self.predictor is not None:
                predicted_feature = self.predictor(ta_feature)
                outputs.update(predicted_feature=predicted_feature)
            
        return outputs

    def update_network(self, num_new_classes) -> None:
        new_ts_net = backbone_dispatch(self.backbone_configs)

        if hasattr(new_ts_net, 'fc'):
            new_ts_net.fc = None
        new_ts_net.to(self.device)
        self.ts_nets.append(new_ts_net)
        if len(self.ts_nets) > 1 and (parameter_count(self.ts_nets[-1]) == parameter_count(self.ts_nets[-2])):
            if self.configs.get('init_from_last'):
                # init from last backbone
                self.ts_nets[-1].load_state_dict(self.ts_nets[-2].state_dict())
            elif self.configs.get('init_from_interpolation'):
                # init from interpolation
                gamma = self.configs['init_interpolation_factor']
                for p_ta, p_ts_old, p_ts_new in zip(self.ta_net.parameters(), self.ts_nets[-2].parameters(), self.ts_nets[-1].parameters()):
                    p_ts_new.data = gamma * p_ts_old.data + (1 - gamma) * p_ta.data
        
        new_dim = new_ts_net.out_dim
        classifier = SimpleLinear(self.feature_dim, self.output_dim + num_new_classes, device=self.device)
        if self.classifier is not None:
            classifier.weight.data[:self.output_dim, :-new_dim] = self.classifier.weight.data
            classifier.weight.data[:self.output_dim, -new_dim:] = 0.
            classifier.bias.data[:self.output_dim] = self.classifier.bias.data
        self.classifier = classifier

        if len(self.ts_nets) > 1:
            self.aux_classifier = SimpleLinear(new_dim, num_new_classes + 1, device=self.device)
        
            if self.predictor is None:
                self.predictor = SimpleLinear(self.ta_feature_dim, self.ta_feature_dim, device=self.device)
            
            if self.ts_attn is None:
                self.ts_attn = TSAttention(new_dim, self.configs['attn_num_heads'], device=self.device)
            else:
                self.ts_attn._reset_parameters()
                
            self.trans_classifier = SimpleLinear(self.ta_net.out_dim, num_new_classes, device=self.device)
    
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
        for m in self.ts_nets[:-1]:
            m.eval()
        return self

    def eval(self):
        return self.train(False)
    
    def get_freezed_copy_ta(self):
        from copy import deepcopy
        ta_net_copy = deepcopy(self.ta_net)
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        from copy import deepcopy
        projector_copy = deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()

    def freeze_old_backbones(self):
        for p in self.ts_nets[:-1].parameters():
            p.requires_grad_(False)
        
        if hasattr(self, 'ts_adapters'):
            for p in self.ts_adapters[:-1].parameters():
                p.requires_grad_(False)
        
        for net in self.ts_nets[:-1]:
            net.eval()


class TSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm_ts = nn.LayerNorm(embed_dim, device=device)
        self.norm_ta = nn.LayerNorm(embed_dim, device=device)

        self.weight_q = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
    
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight_q)
        nn.init.xavier_normal_(self.weight_k_ts)
        nn.init.xavier_normal_(self.weight_k_ta)
        nn.init.xavier_normal_(self.weight_v_ts)
        nn.init.xavier_normal_(self.weight_v_ta)

        self.norm_ta.reset_parameters()
        self.norm_ts.reset_parameters()
    
    def forward(self, ta_feats, ts_feats):
        bs, N, C = ta_feats.shape
        # feats: (bs, N, C)
        ta_feats = self.norm_ta(ta_feats)
        ts_feats = self.norm_ts(ts_feats)

        q = (ts_feats @ self.weight_q).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2) # (bs, H, N, Ch)
        k_ts = (ts_feats @ self.weight_k_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_ta = (ta_feats @ self.weight_k_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ts = (ts_feats @ self.weight_v_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ta = (ta_feats @ self.weight_v_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        feat = nn.functional.scaled_dot_product_attention(q, torch.cat((k_ta, k_ts), dim=2), torch.cat((v_ta, v_ts), dim=2)) # (bs, H, N, Ch) # use default scale

        feat = feat.transpose(1, 2).flatten(2)

        return feat