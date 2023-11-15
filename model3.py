# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim

# 定义输入数据x和输出数据y
# 直接在GPU上构建张量
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=torch.device('cuda:0'))
y = torch.tensor([1, 4, 9, 16, 25], dtype=torch.float32, device=torch.device('cuda:0'))


# 定义多项式回归模型，即y = ax^2 + bx + c
class PolyModel(nn.Module):
    def __init__(self):
        super(PolyModel, self).__init__()
        self.poly = nn.Linear(2, 1)  # 修改输入维度为2

    def forward(self, x):
        # 对输入数据进行特征变换，即增加一个x^2的项
        x = torch.cat([x.unsqueeze(1), x.unsqueeze(1) ** 2], dim=1)
        # 在特征变换后的数据上应用线性层
        out = self.poly(x)
        return out


model = PolyModel().to(torch.device('cuda:0'))  # 将模型移动到GPU上

# 定义损失函数，即预测值和真实值之间的平方差的均值
criterion = nn.MSELoss(reduction='sum')

# 定义优化器，即使用Adam法来更新参数
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义学习率调整策略，即每隔100步将学习率乘以0.9
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


# 定义训练步骤，即计算损失并应用优化器
def train_step(x, y):
    # 将模型设为训练模式
    model.train()
    # 前向传播，得到预测值
    y_pred = model(x)
    # 计算损失
    loss = criterion(y_pred, y)
    # 反向传播，得到梯度
    loss.backward()
    # 应用优化器，更新参数
    optimizer.step()
    # 应用学习率调整策略，更新学习率
    scheduler.step()
    # 清空梯度
    optimizer.zero_grad()
    return loss.item()


# 进行1000次训练迭代
for i in range(1000):
    loss = train_step(x, y)
    print(f"Step {i + 1}: loss = {loss}")

# 使用训练好的模型来预测下一个数字
x_next = torch.tensor([6], dtype=torch.float32, device=torch.device('cuda:0'))
y_next = model(x_next)
print(f"Next number: {y_next.item()}")
