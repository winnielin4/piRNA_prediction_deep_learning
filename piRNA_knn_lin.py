import tensorflow as tf 
import numpy as np

import piRNA_datasets as input_data

piRNA = input_data.read_data_sets()

train_X, train_Y = piRNA.train.next_batch(13810) # 5000 for training (nn candidates)
# test_X, test_Y = mnist.test.next_batch(100)   # 200 for testing
test_X = piRNA.test.images
test_Y = piRNA.test.labels

tra_X = tf.placeholder("float", [None, 175])
te_X = tf.placeholder("float", [175])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(tra_X, tf.negative(te_X))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(test_X)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={tra_X: train_X, te_X: test_X[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(train_Y[nn_index]), \
            "True Class:", np.argmax(test_Y[i]))
        # Calculate accuracy
        if np.argmax(train_Y[nn_index]) == np.argmax(test_Y[i]):
            accuracy += 1./len(test_X)
    print("Done!")
    print("Accuracy:", accuracy)
