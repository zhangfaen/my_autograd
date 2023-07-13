import torch
from mini_autograd import Value

def test_basic1():
    x1 = Value(-4.0, requires_grad=True)
    z1 = 2 * x1 + 2 + x1
    h1 = z1 * z1
    h1.backward()
    print(f"h1 data:{h1.data}, x1 grad:{x1.grad}")

    x2 = torch.Tensor([-4.0]).double()
    x2.requires_grad = True
    z2 = 2 * x2 + 2 + x2
    h2 = z2 * z2 # d_h2 / d_z2 = 2 * z2 * (d_z2 / d_x2) = 2 * z2 * 3 = 6 * 14 = 84
    h2.backward()
    print(f"h2 data:{h2.item()}, x2 grad:{x2.grad.item()}")

    assert h1.data == h2.item()
    assert x1.grad == x2.grad.item()

def test_basic2():
    x1 = Value(3.0, requires_grad=True)
    y1 = Value(2.0, requires_grad=True)
    z1 = x1**2 + x1*y1 + y1 + 1.0
    h1 = z1 * z1
    h1.backward()
    print(f"h1 data:{h1.data}, x1 grad:{x1.grad}, y1 grad:{y1.grad}")

    x2 = torch.Tensor([3.0]).double()
    y2 = torch.Tensor([2.0]).double()
    x2.requires_grad = True
    y2.requires_grad = True
    z2 = x2**2 + x2*y2 + y2 + 1.0
    h2 = z2 * z2
    h2.backward()
    print(f"h2 data:{h2.item()}, x2 grad:{x2.grad.item()}, y2 grad:{y2.grad.item()}")

    assert h2.item() == h1.data
    assert x2.grad.item() == x1.grad and y2.grad.item() == y1.grad


def test_basic3():
    x1 = Value(3.0, requires_grad=True)
    y1 = Value(2.0, requires_grad=True)
    z1 = x1**2 - x1*y1 - y1 + 1.0
    h1 = 1 - z1 * z1
    h1.backward()
    print(f"h1 data:{h1.data}, x1 grad:{x1.grad}, y1 grad:{y1.grad}")

    x2 = torch.Tensor([3.0]).double()
    y2 = torch.Tensor([2.0]).double()
    x2.requires_grad = True
    y2.requires_grad = True
    z2 = x2**2 - x2*y2 - y2 + 1.0
    h2 = 1 - z2 * z2
    h2.backward()
    print(f"h2 data:{h2.item()}, x2 grad:{x2.grad.item()}, y2 grad:{y2.grad.item()}")

    assert h2.item() == h1.data
    assert x2.grad.item() == x1.grad and y2.grad.item() == y1.grad


def test_basic4():
    """ 
    a = 2.0
    b = a * a
    c = 2 * b
    d = 3 * b
    e = c + d

           e
        /    \ 
       c      d
     /   \   /  \
    2      b    3
           |
          a*a
    """
    a1 = Value(1.5, requires_grad=True)
    b1 = a1 * a1
    c1 = 2 * b1
    d1 = 3 * b1
    e1 = c1 + d1
    e1.backward()
    print(f"e1 data:{e1.data}, a1 grad:{a1.grad}")

    a2 = torch.Tensor([1.5]).double()
    a2.requires_grad = True
    b2 = a2 * a2
    c2 = 2 * b2
    d2 = 3 * b2
    e2 = c2 + d2
    e2.backward()
    print(f"e2 data:{e2.item()}, a2 grad:{a2.grad.item()}")
    assert a1.grad == a2.grad.item() and e1.data == e2.item()

def test_basic5():
    """ 
    a = 2.0
    b = a * a
    c = 2 / b
    d = b / 3
    e = c + d

           e
        /    \ 
       c      d
     /   \   /  \
    2      b    3
           |
          a*a
    """
    a1 = Value(1.5, requires_grad=True)
    b1 = a1 * a1
    c1 = 2 / b1
    d1 = b1 / 3
    e1 = c1 + d1
    e1.backward()
    print(f"e1 data:{e1.data}, a1 grad:{a1.grad}")

    a2 = torch.Tensor([1.5]).double()
    a2.requires_grad = True
    b2 = a2 * a2
    c2 = 2 / b2
    d2 = b2 / 3
    e2 = c2 + d2
    e2.backward()
    print(f"e2 data:{e2.item()}, a2 grad:{a2.grad.item()}")

    assert a1.grad == a2.grad.item() and e1.data == e2.item()

def test_basic6():
    a = Value(-4.0, requires_grad=True)
    b = Value(2.0, requires_grad=True)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    ama, bma, gma = a, b, g
    print("----")
    print(a)
    print(b)
    print(g)

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    print(f"gmg.data: {gma.data}, gpt.data.item(): {gpt.data.item()}")
    assert abs(gma.data - gpt.data.item()) < tol
    # backward pass went well
    print(f"amg.grad {ama.grad}, apt.grad.item() {apt.grad.item()}")
    assert abs(ama.grad - apt.grad.item()) < tol
    assert abs(bma.grad - bpt.grad.item()) < tol

def test_basic7():
    x = Value(-4.0, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xma, yma = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert yma.data == ypt.data.item()
    # backward pass went well
    assert xma.grad == xpt.grad.item()


def test_basic8():
    x = Value(2.0, requires_grad=True)
    y = Value(3.0, requires_grad=True)
    z = x**y + y**x + 2
    z.backward()
    xma, yma, zma = x, y, z

    x = torch.Tensor([2.0]).double()
    y = torch.Tensor([3.0]).double()
    x.requires_grad = True
    y.requires_grad = True
    z = x**y + y**x + 2
    z.backward()
    xpt, ypt, zpt = x, y, z
    
    # forward pass went well
    assert zma.data ==  zpt.data.item()
   
    # backward pass went well
    
    assert xma.grad == xpt.grad.item()
    assert yma.grad == ypt.grad.item()


def test_basic9():
    x = Value(0.2, requires_grad=True)
    y = Value(0.8, requires_grad=True)
    z = x**y + y**x + 2
    z = z.sigmoid()
    z.backward()
    xma, yma, zma = x, y, z

    x = torch.Tensor([0.2]).double()
    y = torch.Tensor([0.8]).double()
    x.requires_grad = True
    y.requires_grad = True
    z = x**y + y**x + 2
    z = z.sigmoid()
    z.backward()
    xpt, ypt, zpt = x, y, z
    
    tol = 1e-6

    # forward pass went well
    assert abs(zma.data - zpt.data.item()) < tol
   
    # backward pass went well
    assert abs(xma.grad - xpt.grad.item()) < tol
    assert abs(yma.grad - ypt.grad.item()) < tol


def test_basic10():
    x = Value(0.2, requires_grad=True)
    y = Value(0.8, requires_grad=True)
    z = x.log() + y.log()  
    z = z.sigmoid()
    z.backward()
    xma, yma, zma = x, y, z
    print(x, y, z)

    x = torch.Tensor([0.2]).double()
    y = torch.Tensor([0.8]).double()
    x.requires_grad = True
    y.requires_grad = True
    z = torch.log(x) + torch.log(y)
    z = z.sigmoid()
    z.backward()
    xpt, ypt, zpt = x, y, z
    print(x, y, z)
    
    tol = 1e-6

    # forward pass went well
    assert abs(zma.data - zpt.data.item()) < tol
   
    # backward pass went well
    assert abs(xma.grad - xpt.grad.item()) < tol
    assert abs(yma.grad - ypt.grad.item()) < tol


if __name__ == '__main__':
    test_basic1()
    test_basic2()
    test_basic3()
    test_basic4()
    test_basic5()
    test_basic6()
    test_basic7()
    test_basic8()
    test_basic9()
    test_basic10()


