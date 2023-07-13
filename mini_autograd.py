
from __future__ import annotations
import math

class Value:
    def __init__(self, data: float, requires_grad=False, prev_values: list[Value] = [], name:str = None, op:str = None) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.name = name
        self.op = op
        if self.requires_grad:
            self.grad = 0.0
            self._backward = lambda: None
        self.prev_values = prev_values

    def __repr__(self) -> str:
        return f"data:{'%9.4f' % self.data}, grad:{('%9.4f' % self.grad) if self.requires_grad else '-'}, name:{self.name or ''}, op:{self.op or ''}"

    def __add__(self, other: Value | int | float) -> Value:
        other: Value = other if isinstance(other, Value) else Value(other)
        if other.data == 0 and other.requires_grad == False:
            return self
        out = Value(self.data + other.data,
                    requires_grad=(self.requires_grad or other.requires_grad), prev_values=[self, other], op="+")
        def _backward():
            if self.requires_grad: self.grad += out.grad
            if other.requires_grad: other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Value) -> Value:  # other * self
        return self + other
    
    def __neg__(self) -> Value:
        return self * -1
    
    def __sub__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = self + (-other)
        return out
    
    def __rsub__(self, other) -> Value: # other - self
        return -(self - other)
    
    def __truediv__(self, other) -> Value: # self / other
        return self * (other ** -1)
    
    def __rtruediv__(self, other) -> Value: # other / self
        return other * (self ** -1)

    def __mul__(self, other: Value | float) -> Value:
        other: Value = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), prev_values=[self, other], op="*")
        def _backward():
            if str(out) == 'd: 1.0000, g: 2.0000, n:, o:*':
                print("here")
            if self.requires_grad: self.grad += out.grad * other.data
            if other.requires_grad: other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other: Value) -> Value:  # other * self
        return self * other

    def __pow__(self, other: int | float | Value) -> Value:
        other:Value = other if isinstance(other, Value) else Value(other)
        out_requires_grad = self.requires_grad | other.requires_grad
        out = Value(self.data ** other.data, requires_grad=out_requires_grad, prev_values=[self, other], op="^")
        def _backward():
            if self.requires_grad: self.grad += out.grad * other.data * (self.data**(other.data - 1))
            if other.requires_grad: other.grad += out.grad * math.log(self.data)*(self.data ** other.data)
        out._backward = _backward
        return out
    
    def log(self) -> Value:
        out = Value(math.log(self.data), requires_grad=self.requires_grad, prev_values=[self], op="log")
        def _backward():
            if self.requires_grad: self.grad += out.grad * (1.0 / self.data)
        out._backward = _backward
        return out
    
    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0, self.requires_grad, [self], op="relu")
        def _backward():
            if self.requires_grad: self.grad += (out.grad if self.data > 0 else 0)
        out._backward = _backward
        return out
    
    def sigmoid(self) -> Value:
        return 1.0 / (1.0 + (Value(math.e) ** (-self)))

    def backward(self):
        if self.requires_grad == False:
            return
        self.grad = 1.0
        all_values: list[Value] = []
        visited: set[Value] = set()

        def deep_first(cur: Value):
            if cur in visited:
                return
            visited.add(cur)
            for n in cur.prev_values:
                deep_first(n)
            all_values.append(cur)

        deep_first(self)

        for value in reversed(all_values):
            if value.requires_grad: value._backward()
    