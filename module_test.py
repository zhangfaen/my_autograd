import torch
import random
from mini_autograd import Value
from module import cross_entropy_loss

def test_basic1():

    for c in range(10):
        v = [random.uniform(0, 1.0) for _ in range(10)]
    
        xv = [Value(d, requires_grad=True) for d in v]
        target = [0] * 10
        target[c] = 1.0
        loss = cross_entropy_loss(xv, target)
        loss.backward()
        xvmd, lmd = xv, loss

        xv = torch.Tensor(v).double()
        xv.requires_grad = True
        target = [0] * 10
        target[c] = 1
        loss = torch.nn.functional.cross_entropy(xv, torch.Tensor(target).float())
        loss.backward()
        xvpt, lpt = xv, loss

        tol = 1e-6
        assert abs(lmd.data - lpt.item()) < tol
        for a, b in zip(xvmd, xvpt.grad):
            assert abs(a.grad - b.item()) < tol

if __name__ == '__main__':
    test_basic1()