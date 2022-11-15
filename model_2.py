import os
import pickle
from tensorflow import optimizers
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, Conv2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 生成器读取图像
train_dir = r'C:\深度学习\dataaug\train'
val_dir = r'C:\深度学习\dataset\val'
test_dir = r'C:\深度学习\dataset\test'

train_datagen = ImageDataGenerator(
    rescale=1./255,         # 重放缩因子，数值乘以1.0/255（归一化）
    shear_range=0.2,        # 剪切强度（逆时针方向的剪切变换角度）
    zoom_range=0.2,         # 随机缩放的幅度
    horizontal_flip=True    # 进行随机水平翻转
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)

# 构建网络
model = Sequential()
# 第一段
                 # 第一卷积层，64个大小为5×5的卷积核，步长1，激活函数relu，卷积模式same，输入张量的大小
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=(48, 48, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))      # 第一池化层，池化核大小为2×2，步长2
model.add(BatchNormalization())
model.add(Dropout(0.4))     # 随机丢弃40%的网络连接，防止过拟合
# 第二段
model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
# 第三段
model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())                                  # 过渡层
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))             # 全连接层
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))             # 分类输出层
model.summary()

# 编译
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),  # Adam优化器
              metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,                            # 生成训练集生成器
    steps_per_epoch=243,                        # train_num/batch_size=128
    epochs=40,                                  # 数据迭代轮数
    validation_data=validation_generator,       # 生成验证集生成器
    validation_steps=28                         # valid_num/batch_size=128
)

# 评估模型
test_loss, test_acc = model.evaluate(test_generator, steps=28)
print("test_loss: %.4f - test_acc: %.4f" % (test_loss, test_acc * 100))

# 保存模型
model_json = model.to_json()
with open('myModel_2_json.json', 'w') as json_file:
    json_file.write(model_json) # 保存模型的框架
model.save_weights('myModel_2_weight.h5')   # 只保存了模型的参数，并没有保存模型的图结构，需要再次描述模型结构信息才能加载模型
model.save('myModel_2.h5')  # 保存了模型的图结构和模型的参数，直接使用load_model()方法就可加载模型然后做测试

with open('fit_2_log.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt, 0)

# 绘制训练中的损失曲线和精度曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure("acc")
plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('Accuracy curve')
plt.legend()
plt.savefig('acc_2.jpg')
plt.show()

plt.figure("loss")
plt.plot(epochs, loss, 'r-', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Loss curve')
plt.legend()
plt.savefig('loss_2.jpg')
plt.show()
