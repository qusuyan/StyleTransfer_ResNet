import scipy.io
import scipy.misc

import numpy as np
import tensorflow as tf
import time, sys

from utils import add_noise, open_image, save_image, get_files
from vgg import vgg_model
from transform_net import net
from evaluate import loss_func

import argparse

BATCH_SIZE = [1, 300, 400, 3]

parser = argparse.ArgumentParser(description='Run a style transfer pipeline. ')
parser.add_argument('-m', '--mode', dest='mode', default=0, type=int, required=False, help='0 (default) for style transfer; 1 for training')
parser.add_argument('-c', '--content', dest='content_image', default='img/input.jpg', type=str, required=False, help='input content image')
parser.add_argument('-s', '--style', dest='style_image', default='img/starry_night.jpg', type=str, required=False, help='input style image')
parser.add_argument('-o', '--output', dest='output_dir', default='output', type=str, required=False, help='output directory')
parser.add_argument('-e', '--epochs', dest='epochs', default=1000, type=int, required=False, help='number of iterations')


def main(args): 
    epochs = args.epochs

    file_list = get_files('./data')
    num_samples = len(file_list)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    optimal = sys.float_info.max

    style = np.zeros(BATCH_SIZE, dtype=np.float32)
    style[0] = open_image(args.style_image)

    with tf.Graph().as_default(), tf.Session() as sess:

        X_content = tf.placeholder(tf.float32, shape=BATCH_SIZE, name="X_content")
        X_style = tf.placeholder(tf.float32, shape=BATCH_SIZE, name="X_style")
        generated_image = net(X_content/255.0)

        warmup_loss = tf.reduce_mean(tf.square(generated_image - X_content))
        warmup_step = tf.train.AdamOptimizer(0.001).minimize(warmup_loss)

        loss = loss_func(sess, X_style, X_content, generated_image)
        train_step = tf.train.AdamOptimizer(0.00005).minimize(loss)
        sess.run(tf.global_variables_initializer())

        if args.mode == 0:
            saver = tf.train.Saver()
            saver.restore(sess, './model/transform_net.ckpt')
            eval_batch = np.zeros(BATCH_SIZE, dtype=np.float32)
            eval_batch[0] = open_image(args.content_image)
            start_time = time.time()
            output = sess.run(generated_image, feed_dict={X_style:style, X_content:eval_batch})
            end_time = time.time()
            print("Time: %s" % str(end_time - start_time))
            save_image(args.output_dir + "/generated_image.jpg", output)
            return

        print("Warming up...")
        for epoch in range(20): 
            for i in range(num_samples):
                batch = np.zeros(BATCH_SIZE, dtype=np.float32)
                batch[0] = open_image(file_list[i])

                feed_dict = {
                	X_style:style,
                    X_content:batch
                }

                sess.run(warmup_step, feed_dict = feed_dict)

            eval_batch = np.zeros(BATCH_SIZE, dtype=np.float32)
            eval_batch[0] = open_image(args.content_image)
            eval_loss = sess.run(warmup_loss, feed_dict={X_style:style, X_content:eval_batch})
            print("Epoch: %d, Loss: %f" % (epoch, eval_loss))

        print("Start Training...")
        for epoch in range(args.epochs):
            for i in range(num_samples):
                batch = np.zeros(BATCH_SIZE, dtype=np.float32)
                batch[0] = open_image(file_list[i])

                feed_dict = {
                	X_style:style,
                    X_content:batch
                }

                sess.run(train_step, feed_dict = feed_dict)

            eval_batch = np.zeros(BATCH_SIZE, dtype=np.float32)
            eval_batch[0] = open_image(args.content_image)
            start_time = time.time()
            eval_loss = sess.run(loss, feed_dict={X_style:style, X_content:eval_batch})
            end_time = time.time()
            change_time = end_time - start_time
            print("Epoch: %d, Loss: %f, Time: %s" % (epoch, eval_loss, str(change_time)))
            
            if optimal > eval_loss:
                saver = tf.train.Saver()
                res = saver.save(sess, './model/transform_net.ckpt')
                optimal = eval_loss

            if epoch % 10 == 0:
                output = sess.run(generated_image, feed_dict={X_style:style, X_content:eval_batch})
                save_image(args.output_dir + "/" + str(epoch) + ".jpg", output)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
