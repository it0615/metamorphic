@echo off
mmconvert -sf tensorflow -in model_mnist_tensorflow.ckpt.meta -iw model_mnist_tensorflow.ckpt -df keras --dstNodeName Add_2 -om ../keras/model_mnist_keras.h5
