# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim

# 定义超参数
input_size = 1  # 输入的数字个数
output_size = 1  # 输出的数字个数
degree = 2  # 多项式的次数
learning_rate = 0.01  # 学习率
num_epochs = 100  # 训练的轮数


# 定义一个多项式回归模型
class PolynomialRegression(nn.Module):
    def __init__(self, input_size, output_size, degree):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size * degree, output_size)
        self.degree = degree

    def forward(self, x):
        # 对输入的特征进行多项式变换
        x = torch.cat([torch.pow(x, i) for i in range(1, self.degree + 1)], 1)
        # 通过线性层
        out = self.linear(x)
        return out


# 创建一个模型实例
model = PolynomialRegression(input_size, output_size, degree)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 定义一些训练数据，擅长平方数列
x_train = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
y_train = torch.tensor([[1], [4], [9], [16]], dtype=torch.float)

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_train)
    # 计算损失
    loss = criterion(y_pred, y_train)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# 测试模型
x_test = torch.tensor([[5]], dtype=torch.float)
y_test = model(x_test)
print(f'Input: {x_test}')
print(f'Output: {y_test.item()}')

# 保存模型
model_scripted = torch.jit.script(model)  # 把模型转换为TorchScript
torch.jit.save(model_scripted, "model.pth")  # 保存模型
