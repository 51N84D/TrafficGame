from torch import nn


# Define model
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden_layers,
        num_classes,
        normalize_output=True,
    ):
        super(MLP, self).__init__()
        self.model = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers - 1):
            self.model += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.model += [nn.Linear(hidden_size, num_classes)]
        self.model = nn.Sequential(*self.model)

        self.normalize = nn.Softmax(dim=-1)
        self.normalize_output = normalize_output

    def forward(self, x):
        out = self.model(x)
        if self.normalize_output:
            out = self.normalize(out)
        return out
