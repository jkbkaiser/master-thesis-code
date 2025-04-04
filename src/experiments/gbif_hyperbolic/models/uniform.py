import geoopt
import geoopt.manifolds.stereographic.math as gmath
import torch
import torch.nn as nn

MIN_NORM = 1e-15


class MobiusMLR(nn.Module):
    def __init__(self, in_features, out_features, c=1.0):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = geoopt.PoincareBall(c=c)
        points = torch.randn(out_features, in_features) * 1e-5
        points = gmath.expmap0(points, k=self.ball.k)
        self.p_k = geoopt.ManifoldParameter(points, manifold=self.ball)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        input = input.unsqueeze(-2)
        distance, a_norm = self._dist2plane(x=input, p=self.p_k, a=self.a_k, c=self.ball.c, k=self.ball.k, signed=True)
        result = 2 * a_norm * distance
        return result

    def _dist2plane(self, x, a, p, c, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
        sqrt_c = c**0.5
        minus_p_plus_x = gmath.mobius_add(-p, x, k=k, dim=dim)
        mpx_sqnorm = minus_p_plus_x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        mpx_dot_a = (minus_p_plus_x * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            mpx_dot_a = mpx_dot_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * mpx_dot_a
        denom = (1 - c * mpx_sqnorm) * a_norm
        return gmath.arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c, a_norm


class Uniform(nn.Module):
    def __init__(self, backbone, out_features, architecture, **_):
        super().__init__()
        num_classes = architecture

        self.model = backbone
        self.model.head = nn.Linear(out_features, 64)
        self.ball = geoopt.PoincareBallExact(c=1)
        self.classifier = MobiusMLR(64, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out_feature_euc = self.model(x)
        out_feature_hyp = self.ball.expmap0(out_feature_euc)
        return self.classifier(out_feature_hyp)

    def pred_fn(self, logits):
        pass

    def loss_fn(self, logits, genus_labels, species_labels):
        return self.criterion(logits, species_labels)
