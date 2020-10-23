import yaml
import torch.nn
import collections
from copy import deepcopy
from easydict import EasyDict as edict


class Builder(object):
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            print(e)
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def build_network(architecture, builder=Builder(torch.nn.__dict__)):
    """
    Configuration for feedforward network is list by nature. We can write
    this in simple data structures. In yaml format it can look like:
    .. code-block:: yaml
        network:
            - Conv2d:
                args: [3, 16, 25]
                stride: 1
                padding: 2
            - ReLU:
                inplace: true
            - Conv2d:
                args: [16, 25, 5]
                stride: 1
                padding: 2
    Note, that each layer is a list with a single dict, this is for readability.
    For example, `builder` for the first block is called like this:
    .. code-block:: python
        first_layer = builder("Conv2d", *[3, 16, 25], **{"stride": 1, "padding": 2})
    the simpliest ever builder is just the following function:
    .. code-block:: python
         def build_layer(name, *args, **kwargs):
            return layers_dictionary[name](*args, **kwargs)

    Some more advanced builders catch exceptions and format them in debuggable way or merge
    namespaces for name lookup

    .. code-block:: python

        extended_builder = Builder(torch.nn.__dict__, mynnlib.__dict__)
        net = build_network(network, builder=extended_builder)

    """
    layers = []
    for block in deepcopy(architecture):
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        if "args" in kwargs:
            args = kwargs.pop("args")
        else:
            args = []
        layers.append(builder(name, *args, **kwargs))
    return torch.nn.Sequential(*layers)


def load_yaml(path):
    with open(path, 'r') as f:
        return edict(yaml.safe_load(f))

#
# if __name__ == "__main__":
#     config = load_yaml("../configs/baseline_5.yaml")
#     print(config["network"])
#     builder = Builder(torch.nn.__dict__, src.modules.__dict__)
#     net = build_network(config["network"], builder)
#     print(net)