# 导入TensorFlow库
import tensorflow as tf

# 定义输入数据x
x = [5,25,15,17,18.0]

# 导入SavedModel格式的模型
loaded_model = tf.saved_model.load("my_model")

# 获取模型函数的签名，并使用tf.function装饰器转换为可调用的函数
model_fn = tf.function(loaded_model.signatures["serving_default"])

# 使用导入的模型函数来预测下一个数字
y_next = model_fn(tf.constant(x))["output_0"]
print(f"Next number: {y_next.numpy()}")


# # 导入TensorFlow库
# import tensorflow as tf
#
# # 导入SavedModel格式的模型
# loaded_model = tf.saved_model.load("my_model")
#
# # 打印模型的签名
# print(list(loaded_model.signatures.keys()))
