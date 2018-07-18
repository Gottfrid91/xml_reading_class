# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf
import sys
import PIL as pil
from PIL import Image
import argparse
import re
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()

parser.add_argument('--eval_dir', type=str, default = './output/U-net_seg_train',
                          help = """Directory where to write event logs.""")
parser.add_argument('--eval_data', type = str, default='test',
                           help = """Either 'test' or 'train_eval'.""")
parser.add_argument('--checkpoint_dir', type=str, default= './output/U-net_seg_train',
                          help= """Directory where to read model checkpoints.""")
parser.add_argument('--eval_interval_secs', type= int, default= 60,
                            help="""How often to run the eval.""")
parser.add_argument('--num_examples',type=int, default=1,
                            help="""Number of examples to run.""")
parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')
batch_size = 1
NUM_CLASSES = 2
TOWER_NAME = 'tower'
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32#tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def create_variables(name, shape):
    ''', is_fc_layer=False
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    #set weight initialization according to https://arxiv.org/abs/1505.04597
    initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02)

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.00005)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def conv_relu_layer(input_layer, filter_shape, stride, scope_name):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    filter = create_variables(name=scope_name + '/weights', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.relu(conv_layer)
    return output

def deconv2d(x, W,stride=2):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def u_net(images):
    layers = []
    with tf.variable_scope("u_net", reuse=tf.AUTO_REUSE):

        #Encoder
        conv0_1 = conv_relu_layer(images, [3,3,1,64], stride=1, scope_name="conv0_1")
        _activation_summary(conv0_1)
        layers.append(conv0_1)
        conv0_2 = conv_relu_layer(conv0_1, [3, 3, 64, 64], stride=1, scope_name="conv0_2")
        _activation_summary(conv0_2)
        layers.append(conv0_2)

        max0 = tf.nn.max_pool(conv0_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool0')
        layers.append(max0)

        conv1_1 = conv_relu_layer(max0, [3, 3, 64, 128], stride=1, scope_name="conv1_1")
        _activation_summary(conv1_1)
        layers.append(conv1_1)
        conv1_2 = conv_relu_layer(conv1_1, [3, 3, 128, 128], stride=1, scope_name="conv1_2")
        _activation_summary(conv1_2)
        layers.append(conv1_2)

        max1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        layers.append(max1)

        conv2_1 = conv_relu_layer(max1, [3, 3, 128, 256], stride=1, scope_name="conv2_1")
        _activation_summary(conv2_1)
        layers.append(conv1_1)
        conv2_2 = conv_relu_layer(conv2_1, [3, 3, 256, 256], stride=1, scope_name="conv2_2")
        _activation_summary(conv2_2)
        layers.append(conv2_2)

        max2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        layers.append(max2)

        conv3_1 = conv_relu_layer(max2, [3, 3, 256, 512], stride=1, scope_name="conv3_1")
        _activation_summary(conv3_1)
        layers.append(conv3_1)
        conv3_2 = conv_relu_layer(conv3_1, [3, 3, 512, 512], stride=1, scope_name="conv3_2")
        _activation_summary(conv3_2)
        layers.append(conv3_2)

        max3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
        layers.append(max3)

        conv4_1 = conv_relu_layer(max3, [3, 3, 512, 1024], stride=1, scope_name="conv4_1")
        _activation_summary(conv4_1)
        layers.append(conv4_1)

        conv4_2 = conv_relu_layer(conv4_1, [3, 3, 1024, 1024], stride=1, scope_name="conv4_2")
        _activation_summary(conv4_2)
        layers.append(conv4_2)

        #DECONVOLVING LAYER
        ########################################################################################################
        deconv_filter4 = create_variables(name="deconv_4_filter", shape=[2,2,512,1024])
        up4 = tf.nn.conv2d_transpose(conv4_2,filter = deconv_filter4,output_shape= [batch_size,20,50,512],
                                     strides=[1,2,2,1],padding='SAME',name="test")
        _activation_summary(up4)
        layers.append(up4)
        #concat layer
        concat_4 = tf.concat([up4, conv3_2], 3, name='concat4')
        layers.append(concat_4)
        #
        uconv4_1 = conv_relu_layer(concat_4, [3, 3, 1024, 512], stride=1, scope_name="uconv4_1")
        _activation_summary(uconv4_1)
        layers.append(uconv4_1)

        uconv4_2 = conv_relu_layer(uconv4_1, [3, 3, 512, 512], stride=1, scope_name="uconv4_2")
        _activation_summary(uconv4_2)
        layers.append(uconv4_2)
        ##########################
        deconv_filter3 = create_variables(name="deconv_3_filter" , shape=[2,2,256,512])
        up3 = tf.nn.conv2d_transpose(uconv4_2,filter = deconv_filter3,output_shape= [batch_size,40,100,256],
                                     strides=[1,2,2,1],padding='SAME',name="test")
        layers.append(up3)

        # concat layer
        concat_3 = tf.concat([up3, conv2_2], 3, name='concat3')
        layers.append(concat_3)

        uconv3_1 = conv_relu_layer(concat_3, [3, 3, 512, 256], stride=1, scope_name="uconv3_1")
        _activation_summary(uconv3_1)
        layers.append(uconv3_1)

        uconv3_2 = conv_relu_layer(uconv3_1, [3, 3, 256, 256], stride=1, scope_name="uconv3_2")
        _activation_summary(uconv3_2)
        layers.append(uconv3_2)
        ###########################
        deconv_filter2 = create_variables(name="deconv_2_filter" , shape=[2,2,128,256])
        up2 = tf.nn.conv2d_transpose(uconv3_2,filter = deconv_filter2,output_shape= [batch_size,80,200,128],
                                     strides=[1,2,2,1],padding='SAME',name="up2")
        layers.append(up2)
        # concat layer
        concat_2 = tf.concat([up2, conv1_2], 3, name='concat2')
        layers.append(concat_2)

        uconv2_1 = conv_relu_layer(concat_2, [3, 3, 256, 128], stride=1, scope_name="uconv2_1")
        _activation_summary(uconv2_1)
        layers.append(uconv2_1)

        uconv2_2 = conv_relu_layer(uconv2_1, [3, 3, 128, 128], stride=1, scope_name="uconv2_2")
        _activation_summary(uconv2_2)
        layers.append(uconv2_2)
        #############################
        deconv_filter1 = create_variables(name="deconv_1_filter" , shape=[2,2,64,128])
        up1 = tf.nn.conv2d_transpose(uconv2_2,filter = deconv_filter1,output_shape= [batch_size,160,400,64],
                                     strides=[1,2,2,1],padding='SAME',name="up2")
        layers.append(up1)
        # concat layer
        concat_1 = tf.concat([up1, conv0_2], 3, name='concat1')
        layers.append(concat_1)

        uconv1_1 = conv_relu_layer(concat_1, [3, 3, 128, 64], stride=1, scope_name="uconv1_1")
        _activation_summary(uconv1_1)
        layers.append(uconv1_1)

        uconv1_2 = conv_relu_layer(uconv1_1, [3, 3, 64, 64], stride=1, scope_name="uconv1_2")
        _activation_summary(uconv1_2)
        layers.append(uconv1_2)

        ##############FINAL CONVOLUTION########################

        #TO BE IMPLEMENTED FOR A THE CROSS ENTROPY LOSS
        filter = create_variables(name="uconv0_2" + '/weights', shape=[3, 3, 64, 2])
        uconv0_2 = tf.nn.conv2d(uconv1_2, filter, strides=[1, 1, 1, 1], padding='SAME', name="uconv0_2")

        #uconv0_2 = conv_relu_layer(uconv1_2, [3, 3, 64, 2], stride=1, scope_name="uconv0_2")

        _activation_summary(uconv0_2)
        layers.append(tf.reshape(uconv0_2, (-1, NUM_CLASSES)))

        # output = tf.sigmoid(uconv0_2)
        # _activation_summary(output)
        # layers.append(output)

    return layers[-1]

def generalized_u_net(images):
    layers = []
    with tf.variable_scope("generalized_u_net", reuse=tf.AUTO_REUSE):

        #Encoder
        conv0_1 = conv_relu_layer(images, [3,3,1,64], stride=1, scope_name="conv0_1")
        _activation_summary(conv0_1)
        layers.append(conv0_1)
        conv0_2 = conv_relu_layer(conv0_1, [3, 3, 64, 64], stride=1, scope_name="conv0_2")
        _activation_summary(conv0_2)
        layers.append(conv0_2)

        max0 = tf.nn.max_pool(conv0_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool0')
        layers.append(max0)

        conv1_1 = conv_relu_layer(max0, [3, 3, 64, 128], stride=1, scope_name="conv1_1")
        _activation_summary(conv1_1)
        layers.append(conv1_1)
        conv1_2 = conv_relu_layer(conv1_1, [3, 3, 128, 128], stride=1, scope_name="conv1_2")
        _activation_summary(conv1_2)
        layers.append(conv1_2)

        max1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        layers.append(max1)

        conv2_1 = conv_relu_layer(max1, [3, 3, 128, 256], stride=1, scope_name="conv2_1")
        _activation_summary(conv2_1)
        layers.append(conv1_1)
        conv2_2 = conv_relu_layer(conv2_1, [3, 3, 256, 256], stride=1, scope_name="conv2_2")
        _activation_summary(conv2_2)
        layers.append(conv2_2)

        max2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        layers.append(max2)

        conv3_1 = conv_relu_layer(max2, [3, 3, 256, 512], stride=1, scope_name="conv3_1")
        _activation_summary(conv3_1)
        layers.append(conv3_1)
        conv3_2 = conv_relu_layer(conv3_1, [3, 3, 512, 512], stride=1, scope_name="conv3_2")
        _activation_summary(conv3_2)
        layers.append(conv3_2)

        max3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
        layers.append(max3)

        conv4_1 = conv_relu_layer(max3, [3, 3, 512, 1024], stride=1, scope_name="conv4_1")
        _activation_summary(conv4_1)
        layers.append(conv4_1)

        conv4_2 = conv_relu_layer(conv4_1, [3, 3, 1024, 1024], stride=1, scope_name="conv4_2")
        _activation_summary(conv4_2)
        layers.append(conv4_2)

        max4 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')
        layers.append(max4)

        conv5_1 = conv_relu_layer(max4, [3, 3, 1024, 2048], stride=1, scope_name="conv5_1")
        _activation_summary(conv5_1)
        layers.append(conv5_1)

        ####
        max5 = tf.nn.max_pool(conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        layers.append(max4)

        conv6_1 = conv_relu_layer(max5, [3, 3, 2048, 4096], stride=1, scope_name="conv6_1")
        _activation_summary(conv6_1)
        layers.append(conv6_1)


        #DECONVOLVING LAYER
        ########################################################################################################
        deconv_filter6 = create_variables(name="deconv_6_filter", shape=[2, 2, 2048, 4096])
        up6 = tf.nn.conv2d_transpose(conv6_1, filter=deconv_filter6, output_shape=[batch_size, 10, 30, 2048],
                                     strides=[1, 2, 2, 1], padding='SAME', name="up6")
        _activation_summary(up6)
        layers.append(up6)
        # concat layer
        concat_6 = tf.concat([up6, conv5_1], 3, name='concat6')
        layers.append(concat_6)
        #
        uconv6_1 = conv_relu_layer(concat_6, [3, 3, 4096, 2048], stride=1, scope_name="uconv6_1")
        _activation_summary(uconv6_1)
        layers.append(uconv6_1)

        uconv6_2 = conv_relu_layer(uconv6_1, [3, 3, 2048, 2048], stride=1, scope_name="uconv6_2")
        _activation_summary(uconv6_2)
        layers.append(uconv6_2)


        deconv_filter5 = create_variables(name="deconv_5_filter", shape=[2, 2, 1024, 2048])
        up5 = tf.nn.conv2d_transpose(uconv6_2, filter=deconv_filter5, output_shape=[batch_size, 20, 60, 1024],
                                     strides=[1, 2, 2, 1], padding='SAME', name="test")
        _activation_summary(up5)
        layers.append(up5)
        # concat layer
        concat_5 = tf.concat([up5, conv4_2], 3, name='concat5')
        layers.append(concat_5)
        #
        uconv5_1 = conv_relu_layer(concat_5, [3, 3, 2048, 1024], stride=1, scope_name="uconv5_1")
        _activation_summary(uconv5_1)
        layers.append(uconv5_1)

        uconv5_2 = conv_relu_layer(uconv5_1, [3, 3, 1024, 1024], stride=1, scope_name="uconv5_2")
        _activation_summary(uconv5_2)
        layers.append(uconv5_2)

        ############################
        deconv_filter4 = create_variables(name="deconv_4_filter", shape=[2,2,512,1024])
        up4 = tf.nn.conv2d_transpose(uconv5_2,filter = deconv_filter4,output_shape= [batch_size,40,120,512],
                                     strides=[1,2,2,1],padding='SAME',name="test")
        _activation_summary(up4)
        layers.append(up4)
        #concat layer
        concat_4 = tf.concat([up4, conv3_2], 3, name='concat4')
        layers.append(concat_4)
        #
        uconv4_1 = conv_relu_layer(concat_4, [3, 3, 1024, 512], stride=1, scope_name="uconv4_1")
        _activation_summary(uconv4_1)
        layers.append(uconv4_1)

        uconv4_2 = conv_relu_layer(uconv4_1, [3, 3, 512, 512], stride=1, scope_name="uconv4_2")
        _activation_summary(uconv4_2)
        layers.append(uconv4_2)
        ##########################
        deconv_filter3 = create_variables(name="deconv_3_filter" , shape=[2,2,256,512])
        up3 = tf.nn.conv2d_transpose(uconv4_2,filter = deconv_filter3,output_shape= [batch_size,80,240,256],
                                     strides=[1,2,2,1],padding='SAME',name="test")
        layers.append(up3)

        # concat layer
        concat_3 = tf.concat([up3, conv2_2], 3, name='concat3')
        layers.append(concat_3)

        uconv3_1 = conv_relu_layer(concat_3, [3, 3, 512, 256], stride=1, scope_name="uconv3_1")
        _activation_summary(uconv3_1)
        layers.append(uconv3_1)

        uconv3_2 = conv_relu_layer(uconv3_1, [3, 3, 256, 256], stride=1, scope_name="uconv3_2")
        _activation_summary(uconv3_2)
        layers.append(uconv3_2)
        ###########################
        deconv_filter2 = create_variables(name="deconv_2_filter" , shape=[2,2,128,256])
        up2 = tf.nn.conv2d_transpose(uconv3_2,filter = deconv_filter2,output_shape= [batch_size,160,480,128],
                                     strides=[1,2,2,1],padding='SAME',name="up2")
        layers.append(up2)
        # concat layer
        concat_2 = tf.concat([up2, conv1_2], 3, name='concat2')
        layers.append(concat_2)

        uconv2_1 = conv_relu_layer(concat_2, [3, 3, 256, 128], stride=1, scope_name="uconv2_1")
        _activation_summary(uconv2_1)
        layers.append(uconv2_1)

        uconv2_2 = conv_relu_layer(uconv2_1, [3, 3, 128, 128], stride=1, scope_name="uconv2_2")
        _activation_summary(uconv2_2)
        layers.append(uconv2_2)
        #############################
        deconv_filter1 = create_variables(name="deconv_1_filter" , shape=[2,2,64,128])
        up1 = tf.nn.conv2d_transpose(uconv2_2,filter = deconv_filter1,output_shape= [batch_size,320,960,64],
                                     strides=[1,2,2,1],padding='SAME',name="up2")
        layers.append(up1)
        # concat layer
        concat_1 = tf.concat([up1, conv0_2], 3, name='concat1')
        layers.append(concat_1)

        uconv1_1 = conv_relu_layer(concat_1, [3, 3, 128, 64], stride=1, scope_name="uconv1_1")
        _activation_summary(uconv1_1)
        layers.append(uconv1_1)

        uconv1_2 = conv_relu_layer(uconv1_1, [3, 3, 64, 64], stride=1, scope_name="uconv1_2")
        _activation_summary(uconv1_2)
        layers.append(uconv1_2)

        ##############FINAL CONVOLUTION########################

        #TO BE IMPLEMENTED FOR A THE CROSS ENTROPY LOSS
        filter = create_variables(name="uconv0_2" + '/weights', shape=[3, 3, 64, 2])
        uconv0_2 = tf.nn.conv2d(uconv1_2, filter, strides=[1, 1, 1, 1], padding='SAME', name="uconv0_2")

        #uconv0_2 = conv_relu_layer(uconv1_2, [3, 3, 64, 2], stride=1, scope_name="uconv0_2")

        _activation_summary(uconv0_2)
        layers.append(tf.reshape(uconv0_2, (-1, NUM_CLASSES)))

        #output = tf.sigmoid(uconv0_2)
        #_activation_summary(output)
        #layers.append(output)

    return layers[-1]
def segmentation_loss(logits, labels, class_weights, mask):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size * height * width, num_classes]
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None
        mask: Tensor, weighting bordering pixels extra, [batch_size * height * width,  num_classes]
    Returns:
        segment_loss: Segmentation loss
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
         labels=labels, logits=logits, name='segment_cross_entropy_per_example')

    #mse = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    if class_weights is not None:
        weights = tf.matmul(labels, class_weights, a_is_sparse=True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = tf.multiply(cross_entropy, weights)

    if mask is not None:
        #reshape mask to batch_size * height * width
        mask = tf.reshape(mask, shape = [-1])
        cross_entropy = tf.divide(tf.multiply(cross_entropy, mask, name = "border_masking"), 100)

    segment_loss = tf.reduce_mean(cross_entropy, name='segment_cross_entropy')

    tf.summary.scalar("loss/segmentation", segment_loss)

    return segment_loss

def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """

    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    tf.summary.scalar("loss/weights", l2_loss)

    return l2_loss


def loss(logits, labels, weight_decay_factor, mask=None, class_weights=None):
    """
    Total loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]
        weight_decay_factor: float, factor with which weights are decayed
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None
    Returns:
        total_loss: Segmentation + Classification losses + WeightDecayFactor * L2 loss
    """
    labels = tf.one_hot(tf.cast(tf.reshape(labels, shape=[-1]),tf.int32), depth = NUM_CLASSES)
    segment_loss= segmentation_loss(logits, labels, class_weights, mask)
    total_loss = segment_loss + weight_decay_factor * l2_loss()

    tf.summary.scalar("loss/total", total_loss)

    return total_loss

def dice_coe(output, target,smooth=1e-5):
    '''
    :param output:
    :param target:
    :param smooth:
    :return:
    '''
    target = tf.one_hot(tf.cast(tf.reshape(target, shape=[-1]), tf.int32), depth=2)
    inse = tf.reduce_sum(output * target)
    l = tf.reduce_sum(output * output)
    r = tf.reduce_sum(target * target)
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice

def accuracy(logits, labels):
    """
    Segmentation accuracy:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]
    Returns:
        segmentation_accuracy: Segmentation accuracy
    """

    labels = tf.to_int64(labels)
    labels = tf.reshape(labels, [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
    correct_predictions = tf.equal(predicted_annots, labels)
    segmentation_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    tf.summary.scalar("accuarcy/segmentation", segmentation_accuracy)

    return segmentation_accuracy


def _add_loss_summaries(total_loss):
  """Add summaries for losses in digits-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
