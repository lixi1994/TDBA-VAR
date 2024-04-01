import torch
import torch.nn as nn


class SlowfastWithInternalOutput(nn.Module):
    def __init__(self, NC, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.model.blocks[-1].proj = torch.nn.Linear(in_features=self.model.blocks[-1].proj.in_features, out_features=NC)

    def forward(self, x):
        outputs = []
        for name, value in self.model.named_children():
            for n, layer in value.named_children():
                x = layer(x)
                if n in ['0','1','2', '3', '4']:
                    outputs.append(x)
        # outputs.append(None)
        outputs.append(x)

        return outputs


def slowfast_process_load_state(state_dict):
    state_dict_2 = {}
    for key, value in state_dict.items():
        key_new = 'model.' + key
        state_dict_2[key_new] = value
    return state_dict_2







