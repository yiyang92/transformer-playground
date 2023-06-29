from functools import partial

from torch import Tensor
from torch.nn.functional import gelu, relu, tanh, softplus, silu

from tplayground.utils.constants import Activations


# Gaussian linear unit https://arxiv.org/abs/1606.08415
# f(x) = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
# (x + 0.044715 * torch.pow(x, 3.0))))
_gelu_new = partial(gelu, approximate="tanh")


def _mish(x: Tensor) -> Tensor:
    return x * tanh(softplus(x))


ACTIVATIONS_MAP = {
    Activations.gelu: _gelu_new,
    Activations.relu: relu,
    Activations.mish: _mish,
    Activations.swish: silu
}
