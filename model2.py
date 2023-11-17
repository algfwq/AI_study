# 导入TensorFlow库
import tensorflow as tf

# 定义输入数据x和输出数据y
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
# y = [2, 3, 4, 5, 6]
#
# # 定义模型参数a，b和c，并初始化为随机值
# a = tf.Variable(tf.random.normal(shape=()))
# b = tf.Variable(tf.random.normal(shape=()))
# c = tf.Variable(tf.random.normal(shape=()))
#
# # 定义模型函数，即y = ax^2 + bx + c
# def model(x):
#   return a * x * x + b * x + c

class MyModel(tf.Module):

  def __init__(self):
    super(MyModel, self).__init__()
    self.a = tf.Variable(tf.random.normal(shape=()))
    self.b = tf.Variable(tf.random.normal(shape=()))
    self.c = tf.Variable(tf.random.normal(shape=()))

  # 修改模型函数的签名，使其与输入数据的形状和类型相匹配
  @tf.function(input_signature=[tf.TensorSpec(shape=(5,), dtype=tf.float32)])
  def __call__(self, x):
    return self.a * x * x + self.b * x + self.c

model = MyModel()
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
  gradients = tape.gradient(loss, [model.a, model.b, model.c])
  optimizer.apply_gradients(zip(gradients, [model.a, model.b, model.c]))

# 进行1000次训练迭代
for i in range(100000):
  train_step(x, y)
  print(f"Step {i+1}: a = {model.a.numpy()}, b = {model.b.numpy()}, c = {model.c.numpy()}, loss = {loss_fn(y, model(x)).numpy()}")

# 使用训练好的模型来预测下一个数字
x_next = [6,7,8,9,10]
y_next = model(x_next)
print(f"Next number: {y_next.numpy()}")

# 保存为SavedModel格式
tf.saved_model.save(model, "my_model")
