import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
import argparse


NOISE_RATIO = 0.6
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Vgg 19 model for evaluation

parser = argparse.ArgumentParser(description='Evaluate the style and content difference. ')
parser.add_argument('-c', '--content', dest='content_image', default='./img/content.jpg', type=str, required=False, help='input content image')
parser.add_argument('-s', '--style', dest='style_image', default='./img/style.jpg', type=str, required=False, help='input style image')
parser.add_argument('-g', '--generated', dest='generated_image', default='./img/generated.jpg', type=str, required=True, help='generated image')

content_image = None
style_image = None

IMAGE_WIDTH = None
IMAGE_HEIGHT = None
COLOR_CHANNELS = None

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def load_vgg_model(path):
    
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph

def add_noise(content_image, noise_ratio = NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image


def reshape_and_normalize_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - MEANS
    
    return image


def save_image(path, image):
    image = image + MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

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


def compute_style_cost(sess, model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    
    return J


def main(args): 
    global content_image, style_image
    global IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS

    CONTENT_IMAGE = args.content_image
    STYLE_IMAGE = args.style_image

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    generated_image = scipy.misc.imread(args.generated_image)
    IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS = generated_image.shape
    generated_image = reshape_and_normalize_image(generated_image)

    content_image = scipy.misc.imresize(scipy.misc.imread(CONTENT_IMAGE), (IMAGE_HEIGHT, IMAGE_WIDTH))
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imresize(scipy.misc.imread(STYLE_IMAGE), (IMAGE_HEIGHT, IMAGE_WIDTH))
    style_image = reshape_and_normalize_image(style_image)

    model = load_vgg_model(VGG_MODEL)

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)

    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generated_image))
    
    J_c, J_s = sess.run([J_content, J_style])
    print("Content Loss: " + str(J_c))
    print("Style Loss: " + str(J_s))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)