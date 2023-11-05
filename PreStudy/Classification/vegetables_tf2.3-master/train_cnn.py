# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : train_cnn.py
# @Software: PyCharm
# @Brief   : cnn model to train，code - models ,generated graph - result

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # load train data set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # load test data set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # return train data set ,validation set and class name
    return train_ds, val_ds, class_names


# build CNN model
def model_load(IMG_SHAPE=(224, 224, 3), class_num=20):
    # 搭建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot curve
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_cnn.png', dpi=100)


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    print(class_names)


    # train_ds, val_ds, class_names = data_load("D:/Software/data_set/Food2_train",
    #                                           "D:/Software/data_set/Food2_val", 224, 224, 16)
    print(class_names)

    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("models/cnn_fv.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=5)
