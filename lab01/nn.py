# pylint: disable=missing-function-docstring, missing-class-docstring, missing-module-docstring

import os

from itertools import chain
from abc import abstractmethod, ABC
from typing import Optional, Literal

import numpy as np
from numpy.typing import NDArray, DTypeLike

if os.environ.get("GPU") == "1":
    import cupy as cp

    xp = cp
else:
    xp = np


class Layer(ABC):
    """
    Interface for any differentiable parametrized NDArray function with
    parameters `θ` that takes a single NDArray `x` and returns a single NDArray `y
    = Layer(x; θ)`.
    """

    x: Optional[NDArray]  # Reference to the inputs of the layer
    y: Optional[NDArray]  # Reference to the outputs of the layer

    @abstractmethod
    def reset(self) -> None:
        """Initialize the layer."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> list[NDArray]:
        """Return a list of references to the parameters of the layer."""
        raise NotImplementedError

    @abstractmethod
    def gradients(self) -> list[Optional[NDArray]]:
        """
        Return a list of references to the gradients ∂Loss/∂θ of the loss
        function w.r.t. the parameters, in the same order as the `.parameters()`
        method.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: NDArray, training: bool) -> NDArray:
        """
        Propagate the input `x` forward through the layer and return the output.
        Save the references to the input and output respectively in `self.x` and
        `self.y`.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_y: NDArray) -> NDArray:
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
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def reset(self):
        self.x = None
        self.y = None
        for layer in self.layers:
            layer.reset()

    def parameters(self) -> list[NDArray]:
        return list(chain(*(layer.parameters() for layer in self.layers)))

    def gradients(self) -> list[Optional[NDArray]]:
        return list(chain(*(layer.gradients() for layer in self.layers)))

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        for layer in self.layers:
            x = layer.forward(x, training)
        self.y = x
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y


class Residual(Layer):
    def __init__(self, layer: Layer):
        self.layer: Layer = layer
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.reset()

    def reset(self) -> None:
        self.x = None
        self.y = None
        self.layer.reset()

    def parameters(self) -> list[NDArray]:
        return self.layer.parameters()

    def gradients(self) -> list[Optional[NDArray]]:
        return self.layer.gradients()

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = self.layer.forward(x, training) + x
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        return self.layer.backward(grad_y) + grad_y


class Activation(Layer):
    def reset(self):
        self.x = None
        self.y = None

    def parameters(self) -> list[NDArray]:
        return []

    def gradients(self) -> list[Optional[NDArray]]:
        return []


class Sigmoid(Activation):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert self.y is not None
        return grad_y * (self.y * (1.0 - self.y))


class ReLU(Activation):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = np.maximum(0, x)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert self.y is not None
        return grad_y * (self.y > 0)


class Tanh(Activation):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert self.y is not None
        return grad_y * (1 - self.y**2)


class GELU(Activation):
    def __init__(self):
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = 0.5 * x * (1 + xp.tanh(0.8 * x))
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert self.y is not None
        assert self.x is not None
        a = 0.8
        return grad_y * ((1.0 + xp.tanh(a * self.x)) * (0.5 + a * (self.x - self.y)))


class Dropout(Layer):
    def __init__(self, p: float):
        assert 0 < p <= 1
        self.x: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.p: float = p
        self.mask: Optional[NDArray[np.bool]] = None

    def reset(self):
        self.x = None
        self.y = None
        self.mask = None

    def parameters(self) -> list[NDArray]:
        return []

    def gradients(self) -> list[Optional[NDArray]]:
        return []

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        if training:
            # Save the dropout mask for backward pass
            self.mask = np.random.rand(*x.shape) > self.p
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.p)
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert self.mask is not None
        return grad_y * self.mask


class Linear(Layer):
    def __init__(
        self,
        vsize: int,
        hsize: int,
        init_method: Literal["Xavier", "He"] = "He",
        dtype: DTypeLike = xp.float32,
    ):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.init_method: Literal["Xavier", "He"] = init_method
        self.dtype: DTypeLike = dtype

        # Input and output NDArrays
        self.x: Optional[NDArray]
        self.y: Optional[NDArray]

        # Parameters and gradients
        self.w: NDArray
        self.b: NDArray
        self.grad_w: Optional[NDArray]
        self.grad_b: Optional[NDArray]

        self.reset()

    def reset(self):
        # Input and output reset
        self.x = None
        self.y = None

        # Weights initialization
        match self.init_method:
            case "Xavier":
                scale = xp.sqrt(6 / (self.vsize + self.hsize))
                self.w = xp.random.uniform(-scale, +scale, size=(self.vsize, self.hsize)).astype(self.dtype)
            case "He":
                scale = xp.sqrt(4 / (self.vsize + self.hsize))
                self.w = xp.random.normal(0, scale, size=(self.vsize, self.hsize)).astype(self.dtype)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        self.b = xp.zeros(shape=(self.hsize,), dtype=self.dtype)

        # Gradients initialization
        self.grad_w = None
        self.grad_b = None

    def parameters(self) -> list[NDArray]:
        return [self.w, self.b]

    def gradients(self) -> list[Optional[NDArray]]:
        return [self.grad_w, self.grad_b]

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.x = x
        self.y = self.b + x @ self.w
        return self.y

    def backward(self, grad_y: NDArray) -> NDArray:
        assert self.x is not None
        self.grad_w = self.x.T @ grad_y
        self.grad_b = grad_y.sum(axis=0)
        return grad_y @ self.w.T
