import torch.nn as nn
import torch.nn.init as init


class KerasInitializedLinear(nn.Linear):
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)
