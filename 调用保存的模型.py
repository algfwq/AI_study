# 导入必要的库
import torch

# 加载模型
model = torch.jit.load("model.pth")  # 加载模型
model.eval()  # 把模型设置为评估模式
# 使用模型进行推理
x_test = torch.tensor([[5]], dtype=torch.float)  # 创建一个测试数据
y_test = model(x_test)  # 用模型进行预测
print(f'Input: {x_test}')
print(f'Output: {y_test.item()}')  # 打印预测结果
