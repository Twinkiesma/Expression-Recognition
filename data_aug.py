import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=20,          # 旋转范围
    width_shift_range=0.1,      # 水平平移范围
    height_shift_range=0.1,     # 垂直平移范围
    shear_range=0.1,            # 透视变换的范围
    zoom_range=0.1,             # 缩放范围
    horizontal_flip=True,       # 水平反转
    fill_mode='nearest')

dir = 'C:/深度学习/dataset/train/1'     # 数据增强文件路径
for filename in os.listdir(dir):
    print(filename)
    img = load_img(dir + '/' + filename)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
    # 下面是生产图片的代码
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='C:/深度学习/dataaug/train/1',
                              save_prefix='1',
                              save_format='jpeg'):
        i += 1
        if i > 5:
            break  # 否则生成器会退出循环
