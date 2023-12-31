# 150行代码，写一个超级迷你版的Pytorch：科普深度学习框架

## 100+行代码，先写一个迷你版的微分计算引擎
-------
代码在[mini_autograd.py](mini_autograd.py)模块中. 
- 定义了一个Value类，其内嵌一个Python float值（8字节浮点数）
- Value类实现了： + - * / ** log 等常见运算符
- Value类实现了：运行时构建计算图，实现了前向计算（forward）和后向梯度传播(backward)
- 例如下面这段程序：

```python 
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
    print(a, b, g)
```
运行上述代码，结果如下：
```python
data:  -4.0000, grad: 138.8338, name:, op:
data:   2.0000, grad: 645.5773, name:, op:
data:  24.7041, grad:   1.0000, name:, op:+
```

## 45行代码，实现一个迷你版的支持多个隐藏层的神经网络库（MLP Network）
-------
代码在[module.py](module.py)模块中.
- 定义了一个Module类（模仿Pytorch框架），作为所有自定义Network的基类
- 定义了Linear类，实现一个线性变换层，激活函数可选：sigmoid，relu等非线性激活函数
- 定义了softmax方法和cross entropy loss function


## 用我们自己的微分计算引擎和神经网络库，训练一个超级简单的加法器网络
-------
代码在[demo_adder_net.py](demo_adder_net.py). 训练数据自动生成，4条样例数据如下：
```python
([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], 5) # input中有5个1，target是5
([1, 0, 1, 1, 1, 0, 1, 1, 0, 1], 7) # input中有7个1，target是7
([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 1) # ...
([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10) # ...
```
这个网络极其简单，我们只是演示一下我们自己的微分计算引擎和神经网络库，可以很好的训练这个网络。
运行： `python mini_autograd_test.py`
大约经过500条数据训练后，网络的准确度接近100%


## 用我们自己的微分计算引擎和神经网络库，训练一个超级简单的奇偶辨别网络
-------
代码在[demo_oddeven_net.py](demo_oddeven_net.py). 训练数据自动生成，4条样例数据如下：
```python
([1, 0], [1]) # input是1和0，target是1
([0, 0], [0]) # input是0和0，target是0
([0, 1], [1]) # ...
([1, 1], [0]) # ...
```
网络定义如下：
```python
class OddEvenNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = Linear(shape=(2, 4), activate="sigmoid", name="L1")
        self.layer2 = Linear(shape=(4, 1), activate="sigmoid", name="L2")

    def parameters(self) -> list[Value]:
        return self.layer1.parameters() + self.layer2.parameters()

    def __call__(self, x: list[Value], target) -> Value:
        x = self.layer1(x)
        x = self.layer2(x)
        # for simplicity, we use square error
        loss = sum([(a - b)**2 for a, b in zip(x, target)])
        return loss, x[0].data
```

这个网络极其简单，我们只是演示一下我们自己的微分计算引擎和神经网络库，可以很好的训练这个网络。
运行： `python demo_oddeven_net.py`
大约经过5000条数据训练后，网络的准确度达到100%

## 尝鲜试用
-------
1. 创建一个新的Python环境: `conda create -n mygrad python=3.10`
2. 激活上述环境: `conda activate mygrad`
3. 下载本代码库: `git clone https://github.com/zhangfaen/my_autograd.git`
4. 切换目录: `cd my_autograd`
5. 安装必要的库: `pip install -r requirements.txt` 注：如果你不想运行下面两个测试用例，可以略过此步（也就是说我们的迷你微分计算引擎和迷你神经网络库，只需要Python就能运行）。
6. 运行测试用例1: `python mini_autograd_test.py` 注：此步可选
7. 运行测试用例2: `python module_test.py` 注：此步可选
8. 训练一个极简的加法器神经网络: `python demo_adder_net.py`
9. 训练一个极简的判断奇偶判断网络: `python demo_oddeven_net.py`

## Citation
-------
If you find this repository useful and want to cite it:

> Faen Zhang, my_autograd, GitHub, https://github.com/zhangfaen/my_autograd