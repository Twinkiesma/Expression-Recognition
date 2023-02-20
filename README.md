# 基于卷积神经网络CNN的面部表情识别
<p> 深度学习在计算机视觉上是非常流行的技术，本文选择卷积神经网络 ( CNN ) 作为构建基础创建模型架构，对 Fer2013 数据集中的面部微表情图像进行识别。
  过程划分为三个阶段：图片处理、特征提取、模型识别。图片预处理是在计算特征之前，排除掉跟脸无关的一切干扰，主要有数据增强、归一化等。特征提取
  是通过卷积神经网络模型的计算 ( 卷积核 ) 来提取面部图像相关特征数据，为之后表情识别提供有效的数据特征。</p>

## 开发环境
Anaconda3，PyCharm2021.2.3，python3.6.13，keras2.6.0，tensorflow-gpu2.6.0

## 一、数据预处理
### （一）数据集的介绍
<p> 训练模型使用的是 Kaggle2013 年面部表情识别挑战赛的数据集 Fer2013。它由 35887 张人脸表情图片组成，
  包含训练集 ( Training ) 28709 张，验证集 ( PublicTest ) 和测试集 ( PrivateTest ) 各 3589 张，每张图片是由大小固定为 48×48 
  的灰度图像组成，共有 7 种表情，分别对应于数字标签 0-6，具体表情对应的标签和中英文如下：0 anger 生气；1 disgust 厌恶；
  2 fear 恐惧；3 happy 开心；4 sad 伤心；5 surprised 惊讶；6 normal 中性。</p>
<div align=center> 
  
![1](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/1.jpg)

</div>
数据集下载地址：https://download.csdn.net/download/weixin_48968649/85883963

### （二）数据预处理
数据集并没有直接给出图片，而是将表情、图片数据、用途的数据保存到csv文件中，第一行是表头，说明每列数据的含义，第一列表示表情标签，第二列为原始图片数据，最后一列为用途。
<div align=center> 
  
![2](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/2.jpg)

</div>

### （三）分离数据集
要想让数据集能够为程序所使用，我们需要将数据集分离并转化为程序方便利用的形式。首先要将数据集按照用途分离成三部分，训练集csv文件、验证集csv文件和测试集csv文件
( 参考代码 [data.py](https://github.com/Twinkiesma/Expression-Recognition/blob/master/data.py) ) 接着把csv文件转化成图片文件供我们观察图片
( 参考代码 [img_data.py](https://github.com/Twinkiesma/Expression-Recognition/blob/master/img_data.py) )
<div align=center> 
  
![3](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/3.jpg)

分离前后对比
</div> 
 
### （四）数据增强
<p> 分析训练集后发现，每个类别的训练数据量差别较大，从下图统计的数据能够明显地看出，1 disgust 厌恶的数据量最少，只有436张样本。
  一个好的训练数据集是训练一个良好模型的前提，没有一个比较合理的训练数据就不可能得到一个性能良好的模型，
  因此，在面对一个分布不是很均匀的数据集时，数据增强就显得非常重要了。</p> 
<div align=center> 
  
![4](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/4.jpg)
  
</div> 
<p> 为了防止数据倾斜，我们使用Keras框架中封装的ImageDataGenerator函数，对训练集1 disgust 厌恶 中的样本图片做一些诸如翻转，平移，旋转之类的数据增强操作，
  此函数在设定的参数范围内做随机的变换，大大增多了数据量，使得1 disgust 厌恶 的样本数量从436张增加到了2738张，训练集数据也从原本的28709张增加到了现在的31011张。
  ( 参考代码 <a href="https://github.com/Twinkiesma/Expression-Recognition/blob/master/data_aug.py">data_aug.py</a> ) </p> 
  
## 二、 卷积神经网络模型搭建
### （一）卷积神经网络组成结构
<p> 卷积神经网络主要由输入层、卷积层、激活函数、池化层、全连接层和输出层组成。
  合理的设置上述层结构并在不同层级之间按需进行Dropout、NB等操作才能最终形成一个高效、准确率高的卷积神经网络模型。</p>
<div align=center> 
  
![5](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/5.jpg)

</div>

#### 1、卷积层
<p> 图像实际是由像素构成的，而像素是一连串的数字组成的，所以图像就是由一连串数字构成的矩阵。卷积核是一系列的滤波器，<ins>用来提取某一种特征</ins>。
  一个卷积核一般包括核大小(Kernel Size)、步长(Stride)以及填充步数(Padding)。我们用卷积核来处理一个图片，当图像特征与过滤器表示的特征相似时，
  可以得到一个比较大的值，不相似时，得到的值就比较小。每个卷积核生成一个特征图，这些特征图堆叠起来组成整个卷积层的输出结果。</p>
<div align=center> 
  
![6](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/6.jpg)

</div>
<p> 也可以理解为，CNN的卷积层是指对输入的不同局部的矩阵和卷积核矩阵相同位置做相乘后求和的结果，<ins>卷积的值越大，特征越明显</ins>。</p>
<div align=center> 
  
![7](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/7.gif)

</div>

#### 2、池化层 
<p> 池化层就是<ins>对数据进行压缩</ins>，它是将输入子矩阵的每n×n个元素变成一个元素。常见的池化层思想认为<ins>最大值或者均值</ins>代表了这个局部的特征，
  从局部区域选择最有代表性的像素点数值代替该区域。可以有效的缩小参数矩阵的尺寸，从而减少最后连接层的中的参数数量，也有加快计算速度和防止过拟合的作用。</p>
<div align=center> 
  
![8](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/8.jpg)

</div>

#### 3、全连接层 
<p> 全连接层<ins>将特征图转化为类别输出</ins>。全连接层不止一层，为了防止过拟合会在各全连接层之间引入DropOut操作，能有效控制模型对噪声的敏感度，同时也保留架构的复杂度。
  除此之外，在卷积层和全连接层中间还添加了一层不属于CNN中特有的结构层级的Flatten层过渡，用来将输入“压平”，即把多维数组转换为相同数量的一维向量。</p>
<div align=center> 
  
![9](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/9.jpg)

</div>

### （二）网络模型与训练
<p> 整个卷积神经网络由三个卷积段、三个全连接层、一个分类层组成，每个卷积段包含具有相同卷积操作的一个卷积层和相同池化操作的一个池化层。不同于最后一段卷积，
  前两段卷积均增加了批量标准化操作（Batch Normalization）和Dropout操作。批标准化操作将一个batch中的数据进行标准化处理，使数据尽量落在激活函数梯度较陡的区域避免梯度消失，
  提高模型的泛化性。Dropout操作随机放弃一定概率的节点信息，以放弃部分计算结果的方式防止模型的“过度学习”导致过拟合的发生。</p>
<div align=center> 
  
![10](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/10.jpg)

</div>
<p> 模型的训练时采用的是批处理形式batch_size设为128，数据迭代40轮；所有卷积层均采用5*5的卷积核；采用ReLU激活函数。这里我采用了两种不同的优化器做对比，
  一个是RMSprop优化器学习率设为0.0001 ( 参考代码 <a href="https://github.com/Twinkiesma/Expression-Recognition/blob/master/model.py">model.py</a> )
  另一个是Adam优化器 ( 参考代码 <a href="https://github.com/Twinkiesma/Expression-Recognition/blob/master/model_2.py">model_2.py</a> )</p>
  
### （三）结果分析
<p> 使用RMSprop优化器训练的结果：经过40轮迭代后，训练集的准确率在76%左右，验证集的准确率在60%左右，模型在测试集的准确率为61%左右，损失值为1.1570。</p>
<div align=center> 
  
![11](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/11.jpg)

</div>
<p> 对数据集进行打乱与归一化操作后，使用Adam优化器训练的结果：经过40轮迭代后，模型在测试集的准确率为62%左右，损失值为1.0300。</p>
<div align=center> 
  
![12](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/12.jpg)

</div>
<p> 可以得出对数据集进行打乱与归一化操作对于提高模型准确率的作用并不大，也许是模型参数没有调好，也可能是数据集精度、像素太低及存在错误标签影响了模型的准确度。
  接下来绘制训练中的精度曲线和损失曲线，对比一下两个优化器的不同。</p>
<div align=center> 
  
![13](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/13.jpg)
  
RMSprop优化器的精度曲线和损失曲线
  
![14](https://github.com/Twinkiesma/Expression-Recognition/blob/master/picture/14.jpg)
  
Adam优化器的精度曲线和损失曲线  
</div>
<p> 分析损失曲线，总体验证集的损失值高于训练集的，说明模型完全可以提取有用特征信息，但验证集的数据波动较大并且曲线没有收敛，
  这点在使用RMSprop优化器训练时更为明显。除此之外，我们还可以看到上方两张图的曲线波动比较大，而下方两张图的曲线波动大大减小。
  这说明RMSprop优化器训练的效果更加丰富，也比较杂乱，Adam优化器训练的效果更加平滑，但细节区域不够精细，且速度较RMSprop优化器慢。</p>
