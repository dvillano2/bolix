# placeholder model
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def forward_with_wins(self):
        """assumption is that the below funciton will handle wins,
        and make appropirate logits in those places... for the time
        being lets say make them all 2, so if you're walking and hit all 2's
        (eqivalently a single 2)
        and make appropriate values 1
        might not literally be the forward method"""
        pass
