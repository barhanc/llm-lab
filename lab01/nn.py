# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring
# pylint: disable=invalid-name
import os
from itertools import chain
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray, DTypeLike


type FloatType = np.float32 | np.float64
type BoolType = np.bool_
type IntpType = np.intp

xp = np

# NOTE: This is a hack to allow GPU execution of NumPy code
# fmt:off
if os.environ.get("GPU") == "1":
    try:
        import cupy as cp
        xp = cp
    except ImportError:
        print("Cupy is not available!")
else:
    xp = np
# fmt:on


class Layer(ABC):
    """
    Interface for any differentiable parametrized NDArray function with
    parameters `θ` that takes a single NDArray `x` and returns a single NDArray `y
    = Layer(x; θ)`.
    """

    x: Optional[NDArray[FloatType]]  # Reference to the inputs of the layer
    y: Optional[NDArray[FloatType]]  # Reference to the outputs of the layer

    @abstractmethod
    def parameters(self) -> list[NDArray[FloatType]]:
        """Return a list of references to the parameters of the layer."""
        raise NotImplementedError

    @abstractmethod
    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        """
        Return a list of references to the gradients ∂Loss/∂θ of the loss
        function w.r.t. the parameters, in the same order as the `.parameters()`
        method.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        """
        Propagate the input `x` forward through the layer and return the output.
        Save the references to the input and output respectively in `self.x` and
        `self.y`.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        """
        Given ∂Loss/∂y (`grad_y`):
            * compute ∂Loss/∂x and ∂Loss/∂θ (where θ are the layer's params);
            * return ∂Loss/∂x;

        NOTE: We assume that the layers are connected in a simple path (i.e. the
        computation graph is linear) and thus we don't have to keep and
        accumulate the gradients ∂Loss/∂y in the layer itself, but can instead
        just dynamically pass ∂Loss/∂y while traversing this linear computation
        graph.
        """
        raise NotImplementedError


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers: tuple[Layer, ...] = layers
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return list(chain(*(layer.parameters() for layer in self.layers)))

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return list(chain(*(layer.gradients() for layer in self.layers)))

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        for layer in self.layers:
            x = layer.forward(x, training)
        self.y = x
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y


class Residual(Layer):
    def __init__(self, layer: Layer):
        self.layer: Layer = layer
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return self.layer.parameters()

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return self.layer.gradients()

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = self.layer.forward(x, training) + x
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        return self.layer.backward(grad_y) + grad_y


class Activation(Layer):
    def parameters(self) -> list[NDArray[FloatType]]:
        return []

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return []


class GELU(Activation):
    def __init__(self):
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        self.y = 0.5 * x * (1 + xp.tanh(0.8 * x))
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.y is not None
        assert self.x is not None
        return grad_y * ((1.0 + xp.tanh(0.8 * self.x)) * (0.5 + 0.8 * (self.x - self.y)))


class Dropout(Layer):
    def __init__(self, p: float):
        assert 0.0 < p <= 1.0
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None
        self.p: float = p
        self.mask: Optional[NDArray[BoolType]] = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return []

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return []

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        self.x = x
        if training:
            # Save the dropout mask for backward pass
            self.mask = xp.random.rand(*x.shape) > self.p
            self.y = x * self.mask / (1 - self.p)
        else:
            self.y = x
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.mask is not None
        return grad_y * self.mask


class Linear(Layer):
    def __init__(self, vsize: int, hsize: int, bias: bool = True, dtype: DTypeLike = xp.float32):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.dtype: DTypeLike = dtype
        self.bias: bool = bias

        # Input and output NDArrays
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

        # Parameters and gradients
        scale = xp.sqrt(4 / (self.vsize + self.hsize))
        self.w: NDArray[FloatType] = xp.random.normal(0, scale, size=(self.vsize, self.hsize)).astype(self.dtype)
        self.b: NDArray[FloatType] = xp.zeros(shape=(self.hsize,), dtype=self.dtype)

        self.grad_w: Optional[NDArray[FloatType]] = None
        self.grad_b: Optional[NDArray[FloatType]] = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return [self.w, self.b] if self.bias else [self.w]

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return [self.grad_w, self.grad_b] if self.bias else [self.grad_w]

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        # Inp shape `(..., vsize)`
        # Out shape `(..., hsize)`
        self.x = x
        self.y = self.b + x @ self.w
        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        assert self.x is not None
        # Inp shape `(..., hsize)`
        # Out shape `(..., vsize)`
        self.grad_w = self.x.reshape(-1, self.vsize).T @ grad_y.reshape(-1, self.hsize)
        self.grad_b = grad_y.reshape(-1, self.hsize).sum(axis=0)
        return grad_y @ self.w.T


def softmax(x: NDArray[FloatType], axis: int) -> NDArray[FloatType]:
    m = x.max(axis=axis, keepdims=True)
    y: NDArray[FloatType] = xp.exp(x - m)
    return y / y.sum(axis=axis, keepdims=True)


class MultiHeadSelfAttention(Layer):
    def __init__(
        self,
        num_heads: int,
        dim_model: int,
        dropout: float,
        dtype: DTypeLike = xp.float32,
    ):
        assert dim_model % num_heads == 0
        assert 0.0 < dropout < 1.0

        self.num_heads: int = num_heads
        self.dim_model: int = dim_model
        self.dim_head: int = dim_model // num_heads
        self.dtype: DTypeLike = dtype

        # Input and output NDArrays
        self.x: Optional[NDArray[FloatType]] = None
        self.y: Optional[NDArray[FloatType]] = None

        # Parameters and gradients
        scale = xp.sqrt(4 / (self.dim_model + self.dim_head))
        size = (self.num_heads, self.dim_model, self.dim_head)

        self.Wq: NDArray[FloatType] = xp.random.normal(0, scale, size).astype(self.dtype)
        self.Wk: NDArray[FloatType] = xp.random.normal(0, scale, size).astype(self.dtype)
        self.Wv: NDArray[FloatType] = xp.random.normal(0, scale, size).astype(self.dtype)

        self.map_out = Sequential(Linear(dim_model, dim_model), Dropout(dropout))

        self.grad_Wq: Optional[NDArray[FloatType]] = None
        self.grad_Wk: Optional[NDArray[FloatType]] = None
        self.grad_Wv: Optional[NDArray[FloatType]] = None

    def parameters(self) -> list[NDArray[FloatType]]:
        return [self.Wq, self.Wk, self.Wv] + self.map_out.parameters()

    def gradients(self) -> list[Optional[NDArray[FloatType]]]:
        return [self.grad_Wq, self.grad_Wk, self.grad_Wv] + self.map_out.gradients()

    def forward(self, x: NDArray[FloatType], training: bool) -> NDArray[FloatType]:
        assert x.ndim == 3
        batch_size, seq_len, _ = x.shape

        # Shape `(batch_size, seq_len, dim_model)`
        self.x = x

        # Shape `(batch_size, 1, seq_len, dim_model)`
        x = x[:, None, :, :]
        # Shape `(batch_size, num_heads, seq_len, dim_head)`
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv
        # Shape `(batch_size, num_heads, seq_len, seq_len)`
        scores = softmax(Q @ K.transpose(0, 1, 3, 2) / self.dim_head**0.5, axis=-1)
        # Shape `(batch_size, num_heads, seq_len, dim_head)`
        self.y = scores @ V
        # Shape `(batch_size, seq_len, num_heads, dim_head)`
        self.y = self.y.transpose(0, 2, 1, 3)
        # Shape `(batch_size, seq_len, dim_model)`
        self.y = self.y.reshape(batch_size, seq_len, -1)
        # Shape `(batch_size, seq_len, dim_model)`
        self.y = self.map_out.forward(self.y, training)

        return self.y

    def backward(self, grad_y: NDArray[FloatType]) -> NDArray[FloatType]:
        batch_size, seq_len, _ = grad_y.shape

        grad_x = self.map_out.backward(grad_y)
        grad_x = grad_x.reshape(batch_size, seq_len, self.num_heads, self.dim_head)
        grad_x = grad_x.transpose(0, 2, 1, 3)


class Embedding(Layer): ...


class LayerNorm(Layer): ...


class Optimizer(ABC):
    # References to the parameters of the model
    parameters: list[NDArray[FloatType]]

    @abstractmethod
    def __init__(self, parameters: list[NDArray[FloatType]], *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, gradients: list[NDArray[FloatType]]) -> None:
        """
        Given the list of gradfients ∂Loss/∂θ of the loss function w.r.t. the
        parameters in the same order as in the `self.parameters` list, apply the
        gradients and advance the optimizer.
        """
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(
        self,
        params: list[NDArray[FloatType]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.lr: float = lr
        self.eps: float = eps
        self.beta1: float = betas[0]
        self.beta2: float = betas[1]
        self.t: int = 0

        self.params: list[NDArray[FloatType]] = params
        self.means: list[NDArray[FloatType]] = [xp.zeros(p.shape, p.dtype) for p in self.params]
        self.vars: list[NDArray[FloatType]] = [xp.zeros(p.shape, p.dtype) for p in self.params]

    def apply(self, gradients: list[NDArray[FloatType]]):
        self.t += 1
        for p, m_p, v_p, grad_p in zip(self.params, self.means, self.vars, gradients):
            m_p *= self.beta1
            m_p += (1 - self.beta1) * grad_p
            v_p *= self.beta2
            v_p += (1 - self.beta2) * grad_p**2

            mhat_p = m_p / (1 - self.beta1**self.t)
            vhat_p = v_p / (1 - self.beta2**self.t)

            p -= self.lr * mhat_p / (self.eps + vhat_p**0.5)
