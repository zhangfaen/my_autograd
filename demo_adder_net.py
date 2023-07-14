import random
from module import Module
from module import Linear
from mini_autograd import Value

random.seed(1337)

def make_data(data_size):
    res = []
    for _ in range(data_size):
        one_count = random.randint(1, 10)
        data = [1] * one_count + [0]*(10 - one_count)
        random.shuffle(data)
        res.append((data, one_count))
    return res

train_data = make_data(500)
eval_data = make_data(100)

for i in [random.randint(0, len(train_data) -1) for j in range(4)]:
    print(train_data[i])

class AdderNet(Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = Linear(shape=(10, 1), activate="", name="L1" )

    def parameters(self) -> list[Value]:
        return self.layer1.parameters() 
            
    def __call__(self, x, target):
        x = self.layer1(x)
        loss = (x[0] - target) ** 2 
        return loss, x[0].data

adder_net = AdderNet()

print("Before training, all parameters are:")
print("\n".join([str(p) for p in adder_net.parameters()]))

for i, td in enumerate(train_data):
    loss, _ = adder_net(td[0], td[1]) # data is tuple like (10_Values, label_of_int)
    adder_net.zero_grad()
    learning_rate = 0.1
    loss.backward()
    
    for p in adder_net.parameters():
        if p.requires_grad:
            p.data -= learning_rate * p.grad  
            
    if i % 50 == 0:
        eval_loss = 0.0
        mean_error = 0.0
        for ed in eval_data:
            temp_loss, temp_logis = adder_net(ed[0], ed[1])
            eval_loss += temp_loss.data
            mean_error += abs(temp_logis - ed[1])
        print(f"After training {i} data samples, mean error:{mean_error / len(eval_data)}, eval loss : {eval_loss}")

print("After training, all parameters are:")
print("\n".join([str(p) for p in adder_net.parameters()]))