# Built on top of the resnet model in CNTK repository
# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/ResNet/Python/resnet_models.py
#
# === Original Copyright ======================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, sigmoid
import cntk as C

#
# Resnet building blocks
#
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init)
    return relu(r)

def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters)
    p  = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
    c1 = conv_bn_relu(input, (3,3), num_filters, strides)
    c2 = conv_bn(c1, (3,3), num_filters)
    s  = conv_bn(input, (1,1), num_filters, strides)
    p  = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters):
    assert (num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_basic(l, num_filters)
    return l

def UpSampling2D(x):
    xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    xx = C.splice(xr, xr, axis=-1) # axis=-1 refers to the last axis
    xy = C.splice(xx, xx, axis=-3) # axis=-3 refers to the middle axis
    r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))

    return r

def UpSampling2DPower(x, k_power):
    for i in range(k_power):
        x = UpSampling2D(x)

    return x

def OneByOneConvAndUpSample(x, k_power, num_channels):
    x = Convolution((1, 1), num_channels, init=he_normal(), activation=relu, pad=True)(x)
    x = UpSampling2DPower(x, k_power)

    return x

def dice_coefficient(x, y):
    # average of per-channel dice coefficient
    # global dice coefificnet doesn't work as class with larger region dominates the metrics
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    intersection = C.reduce_sum(x * y, axis=(1,2))

    return C.reduce_mean(2.0 * intersection / (C.reduce_sum(x, axis=(1,2)) + C.reduce_sum(y, axis=(1,2)) + 1.0))

#
# Defines the fully convolutional models for image segmentation
#
def create_model(input, num_classes):
    c_map = [16, 32, 64]
    num_stack_layers = 3

    conv = conv_bn_relu(input, (3,3), c_map[0])
    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])

    up_r1 = OneByOneConvAndUpSample(r1, 0, num_classes)
    up_r2_2 = OneByOneConvAndUpSample(r2_2, 1, num_classes)
    up_r3_2 = OneByOneConvAndUpSample(r3_2, 2, num_classes)

    merged = C.splice(up_r1, up_r3_2, up_r2_2, axis=0)

    resnet_fcn_out = Convolution((1, 1), num_classes, init=he_normal(), activation=sigmoid, pad=True)(merged)

    return resnet_fcn_out
