# Author: Weiran Lin

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np

import piRNA_datasets as input_data
import cnn_lin_model
import complicated_tensorflow_model as ctm

import tensorflow as tf


FLAGS = None

LOGS_DIRECTORY = "logs/train"
TOTAL_BATCH = 100000




def main(_):
    # Import data
    piRNA = input_data.read_data_sets()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 175])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Build the graph for the deep net
    # y_conv, keep_prob = ctm.deepnn(x)
    y_conv, keep_prob = cnn_lin_model.CNN(x)


    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', cross_entropy)


    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

        max_acc = 0;

        # shuffle the training set
        # np.random.shuffle(piRNA.train);
        
        for i in range(TOTAL_BATCH):
            batch = piRNA.train.next_batch(50)
            
            step, training_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op], feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})            
            
            # Write logs at every iteration
            summary_writer.add_summary(summary, i)
            
            # print out results
            if i % 50 == 0:
                print('step %d, training accuracy %g' % (i, training_accuracy))
            if i % 1000 == 0:
                print('test accuracy %g' % accuracy.eval(feed_dict={x: piRNA.test.images, y_:piRNA.test.labels, keep_prob: 1.0}))
        
        # Test Set
        print('test accuracy %g' % accuracy.eval(feed_dict={x: piRNA.test.images, y_:piRNA.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
