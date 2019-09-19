import tensorflow as tf
import numpy as np

from utils import normalize
from vgg import vgg_model

VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Vgg 19 model for evaluation

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

CONTENT_LAYER = ('conv4_2')

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C, (-1, n_C))
    a_G_unrolled = tf.reshape(a_G, (-1, n_C))
    
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled) ** 2) / 4 / n_H / n_W / n_C / m
    
    return J_content


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, (-1, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, (-1, n_C)))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum((GS - GG) ** 2) / 4 / n_C ** 2 / n_W ** 2 / n_H ** 2
       
    return J_style_layer


def compute_style_cost(sess, STYLE_LAYERS, style_model, generated_model):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        a_S = style_model[layer_name]
        a_G = generated_model[layer_name]
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 40, beta = 10):
    J = alpha * J_content + beta * J_style
    
    return J

def loss_func(sess, style_img, content_img, generated_img):
    content_image = normalize(content_img)
    style_image = normalize(style_img)
    generated_image = normalize(generated_img)

    content_model = vgg_model(VGG_MODEL, content_image)
    a_C = content_model[CONTENT_LAYER]

    generated_model = vgg_model(VGG_MODEL, generated_image)
    a_G = generated_model[CONTENT_LAYER]

    J_content = compute_content_cost(a_C, a_G)

    style_model = vgg_model(VGG_MODEL, style_image)
    J_style = compute_style_cost(sess, STYLE_LAYERS, style_model, generated_model)

    J = total_cost(J_content, J_style)

    return J


