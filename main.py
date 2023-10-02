# 导入TensorFlow库
import tensorflow as tf

# 定义输入数据x和输出数据y
x = [1, 2, 3, 4, 5, 6, 7]
y = [2, 4, 6, 8, 10, 12, 14]

# 定义模型参数a和b，并初始化为随机值
a = tf.Variable(tf.random.normal(shape=()))
b = tf.Variable(tf.random.normal(shape=()))

# 定义模型函数，即y = ax + b
def model(x):
  return a * x + b

# 定义损失函数，即预测值和真实值之间的平方差的均值
def loss_fn(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_true - y_pred))

# 定义优化器，即使用Adam法来更新参数
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# 定义训练步骤，即计算梯度并应用优化器
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = loss_fn(y, y_pred)
  gradients = tape.gradient(loss, [a, b])
  optimizer.apply_gradients(zip(gradients, [a, b]))

# 进行1000次训练迭代
for i in range(10000):
  train_step(x, y)
  print(f"Step {i+1}: a = {a.numpy()}, b = {b.numpy()}, loss = {loss_fn(y, model(x)).numpy()}")

# 使用训练好的模型来预测下一个数字
x_next = 1024
y_next = model(x_next)
print(f"Next number: {y_next.numpy()}")
print(y_next.numpy())
