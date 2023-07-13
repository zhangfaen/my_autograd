from mini_autograd import Value 
import random
import math

class Module:
    def parameters(self) -> list[Value]:
        """ Return all trainable parameters, i.e. list of Values whose requires_grad is True"""
        return []
    
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0 if p.requires_grad else None

class Linear(Module):
    def __init__(self, shape: tuple[int, int], activate="relu", name: str="") -> None:
        super().__init__()
        self.shape = shape
        self.name = name
        self.activate = activate
        self.w = [[Value(random.uniform(-1, 1), requires_grad=True, name=f"{name}-w{j}{i}", op="leaf") for j in range(shape[1])] for i in range(shape[0])]
        self.b = [Value(random.uniform(-1, 1), requires_grad=True, name=f"{name}-b0{i}", op="leaf") for i in range(shape[1])]

    def __call__(self, x: list[Value]) -> list[Value]:
        res = [sum(a * b for a, b in zip(x, w_col)) for w_col in zip(*self.w)]
        if self.activate == 'sigmoid':
            res = [(a + b).sigmoid() for a, b in zip(res, self.b)]
        elif self.activate == 'relu':
            res = [(a + b).relu() for a, b in zip(res, self.b)]
        else:
            res = [(a + b) for a, b in zip(res, self.b)]
        return res
    
    def parameters(self) -> list[Value]:
        return [p for r in self.w for p in r] + self.b

def softmax(x: list[Value]) -> list[Value]:
    ev = Value(math.e, False)
    res = [ev**a for a in x]
    total = sum(res)
    return [a/total for a in res]

def cross_entropy_loss(pred: list[Value], target: list[float]) -> Value:
    pred = softmax(pred)
    loss = sum([-pp.log() * pt for pp, pt in zip(pred, target)])
    return loss
    

