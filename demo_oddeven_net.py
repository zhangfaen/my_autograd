import random
from module import Module
from module import Linear
from mini_autograd import Value

random.seed(1337)

# make data, 计算有奇数个1还是偶数个1
train_data = [([1, 0], [1]), ([1, 1], [0]), ([0, 0], [0]), ([0, 1], [1])] * 2000
eval_data = [([1, 0], [1]), ([1, 1], [0]), ([0, 0], [0]), ([0, 1], [1])] * 100
random.shuffle(train_data)
random.shuffle(eval_data)

for i in [random.randint(0, len(train_data) -1) for j in range(4)]:
    print(train_data[i])

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

oddeven_net = OddEvenNet()

for i, ed in  enumerate(train_data):
    loss, _ = oddeven_net(ed[0], ed[1]) # data is tuple like ([0, 1], [1]), ([0, 0], [0]), ([1, 1], [0]), ([0, 1], [1])
    oddeven_net.zero_grad()
    learning_rate = 0.1
    loss.backward()
    for p in oddeven_net.parameters():
        if p.requires_grad:
            p.data -= learning_rate * p.grad  
    if i % 500 == 0:
        eval_loss = 0
        correct_count, wrong_count = 0, 0
        for ed in eval_data:
            temp_loss, logis = oddeven_net(ed[0], ed[1])
            if (ed[1][0] == 1 and logis > 0.5) or (ed[1][0] == 0 and logis < 0.5):
                correct_count += 1
            else:
                wrong_count += 1
            eval_loss += temp_loss.data
        print(f"After training {i} data samples, correct:{correct_count}, wrong:{wrong_count} acc:{1.0*correct_count/(correct_count + wrong_count)} eval loss : {eval_loss}")

