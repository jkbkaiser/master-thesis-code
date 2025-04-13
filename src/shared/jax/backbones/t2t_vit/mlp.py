from flax import nnx


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.act = act_layer
        self.fc2 = nnx.Linear(hidden_features, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

