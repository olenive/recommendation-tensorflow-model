import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.metrics import accuracy_score

from core import helpers as hlp


DIR_TENSORBOARD_OUTPUT = "tensorboard_output"
TENSORBOARD_REPORT_EPOCH_FREQUENCY = 10


def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    weight = tf.get_variable(name, initializer=initial)
    hlp.attach_summaries_(weight, name=name)
    return weight


def bias_variable(shape, name="b"):
    initial = tf.constant(0.001 * np.random.rand(), shape=shape)
    bias = tf.get_variable(name, initializer=initial)
    hlp.attach_summaries_(bias, name=name)
    return bias


def create_dense_layer(input, num_inputs, num_outputs, name="dense"):
    with tf.variable_scope(name):
        W = weight_variable([num_inputs, num_outputs])
        b = bias_variable([num_outputs])
        input = tf.reshape(input, [-1, num_inputs])
        return tf.nn.sigmoid(tf.matmul(input, W) + b)


def loss_cross_entropy(classification_labels, network_output):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=classification_labels, logits=network_output)
    )
    return loss


def create_simple_network(input, num_inputs=10, num_outputs=1000, name="single-layer-network"):
    with tf.variable_scope(name):
        dense = create_dense_layer(input, num_inputs, num_outputs, name="1-dense")
        return dense


def train_network(num_epochs, train_books, test_books, train_chars, test_chars,
                  tensorboard_output=DIR_TENSORBOARD_OUTPUT,
                  summary_frequency=TENSORBOARD_REPORT_EPOCH_FREQUENCY):
    ## Declare placeholders and create computation graph
    x = tf.placeholder(tf.float32, shape=[None, 10], name="input_characteristics")
    y_out = create_simple_network(x)
    y_ = tf.placeholder(tf.float32, shape=[None, 1000], name="known_books")

    ## Create loss calculation operation and train operation
    with tf.name_scope("train"):
        loss = loss_cross_entropy(y_, y_out)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope("accuracy"):
        op_accuracy = tf.metrics.accuracy(y_, y_out)
        op_summary_accuracy = tf.summary.scalar('accuracy', op_accuracy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Initialise summary writer with graph used in the current session
        writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
        for epoch in range(num_epochs):
            train_step.run(feed_dict={x: train_chars, y_: train_books})

            ## Write tensorboard report data and console output
            if epoch % TENSORBOARD_REPORT_EPOCH_FREQUENCY == 0:
                # Rather than running and adding all the summaries one at a time; merge, run and add in one go
                report_dict = {x: train_chars, y_: train_books}
                writer.add_summary(tf.summary.merge_all().eval(feed_dict=report_dict), epoch)
                # Output accuracy to console
                train_accuracy = op_accuracy.eval(feed_dict={x: train_chars, y_: train_books})
                print('epoch ' + epoch + ', accuracy on training data ' + train_accuracy)
                test_accuracy = op_accuracy.eval(feed_dict={x: test_chars, y_: test_books})
                print('epoch ' + epoch + ', accuracy on training data ' + test_accuracy)

        ## Evaluate network accuracy on the test data set
        print("Final accuracy on test data set: " +
              op_accuracy.eval(feed_dict={x: test_chars, y_: test_books}))
        
        return y_out, writer






def main():
    args = hlp.interpret_command_line_arguments()
    np.random.seed(args.seed)  # Set rng seed to make result reproducible.

    if args.train:
        print('Training network using characteristics from ' + args.characteristics +
              ' and books from ' + args.books)

        # Load data from files
        user_chars = hlp.read_csv_user_chars(args.characteristics)
        user_books = hlp.read_csv_user_books(args.books)

        train_books, test_books, train_chars, test_chars = hlp.create_training_and_testing_sets(
            user_chars, user_books, args.test_set_length
        )



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Initialise summary writer with graph used in the current session
        writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
        for epoch in range(num_cnn_epochs):
            batch = training_data.next_batch(batch_size_cnn)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Initialise summary writer with graph used in the current session
        writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
        for epoch in range(num_cnn_epochs):
            batch = training_data.next_batch(batch_size_cnn)



    elif args.recommend:
        print('Generating recommendations for users from ' + args.characteristics)

if __name__ == "__main__":
    main()
