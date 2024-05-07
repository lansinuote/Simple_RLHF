import torch


class Lora(torch.nn.Module):

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

        self.lora_A = torch.nn.Parameter(
            torch.randn(linear.in_features, 128) * 0.1)
        self.lora_B = torch.nn.Parameter(torch.zeros(128, linear.out_features))

        self.linear.weight.requires_grad = False

    def forward(self, x):
        y_linear = self.linear(x)
        y_lora = x.matmul(self.lora_A).matmul(self.lora_B)
        return y_linear + y_lora / 128


def get_layer(model, name):
    layer = model
    for i in name.split('.'):
        layer = getattr(layer, i)
    return layer


def set_layer(model, name, layer):
    name = name.split('.')
    name_father = '.'.join(name[:-1])
    name_node = name[-1]

    layer_father = get_layer(model, name_father)
    layer_father.__setattr__(name_node, layer)


def insert(model):
    for name, layer in model.named_modules():
        if 'decoder.layers.' not in name:
            continue

        if not isinstance(layer, torch.nn.Linear):
            continue

        set_layer(model, name, Lora(layer))


def merge(model):
    for name, layer in model.named_modules():
        if not isinstance(layer, Lora):
            continue

        linear = layer.linear
        linear.weight.data += layer.lora_A.matmul(layer.lora_B).t() / 128

        set_layer(model, name, linear)


def count_params(model):
    count_all = [i.numel() for i in model.parameters()]
    count_all = sum(count_all) / 1_0000_0000

    count_require = [i.numel() for i in model.parameters() if i.requires_grad]
    count_require = sum(count_require) / 1_0000_0000

    ratio = count_require / count_all

    print({
        'count_require': count_require,
        'count_all': count_all,
        'ratio': ratio
    })