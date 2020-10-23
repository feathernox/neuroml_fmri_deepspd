import torch
from collections import defaultdict
from src.modules.basic import BiMap


class SingularValuesHook():
    def __init__(self, model, only_range=False):
        self.model = model
        self.only_range = only_range

        self.singular_values = None
        self.clear_singular_values()
        self.hooks = []
        self.register_hooks()

    def _get_singular_values(self, input_tensor):
        _, input_sv, _ = torch.svd(input_tensor, compute_uv=False)
        if self.only_range:
            input_sv = {
                'min': input_sv[..., -1].detach().cpu().numpy(),
                'max': input_sv[..., 0].detach().cpu().numpy()
            }
        else:
            input_sv = input_sv.detach().cpu().numpy()
        return input_sv

    def hook_fn(self, module, input, output):
        input_sv = self._get_singular_values(input[0])
        output_sv = self._get_singular_values(output)
        self.singular_values[module]['input'].append(input_sv)
        self.singular_values[module]['output'].append(output_sv)

    def register_hooks(self):
        self.close()
        self.hooks = []
        for name, layer in self.model._modules.items():
            if isinstance(layer, BiMap):
                self.hooks.append(layer.register_forward_hook(self.hook_fn))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def clear_singular_values(self):
        self.singular_values = defaultdict(lambda: {'input': [], 'output': []})
