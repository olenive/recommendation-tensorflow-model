import numpy as np
import tensorflow as tf

from core import common as com
from core import helpers as hlp


DIR_TENSORBOARD_OUTPUT = "tensorboard_output"
TENSORBOARD_REPORT_EPOCH_FREQUENCY = 100


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


def loss_function(classification_labels, network_output):
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=classification_labels, logits=network_output)
    )
    return loss


def create_simple_network(input, num_inputs=10, num_outputs=1000, name="single-layer-network"):
    with tf.variable_scope(name):
        dense1 = create_dense_layer(input, num_inputs, 100, name="1-dense")
        dense_final = create_dense_layer(dense1, 100, num_outputs, name="final-dense")
        return dense_final


def train_network(num_epochs, train_books, test_books, train_chars, test_chars, model_path,
                  tensorboard_output=DIR_TENSORBOARD_OUTPUT,
                  summary_frequency=TENSORBOARD_REPORT_EPOCH_FREQUENCY):

    ## Declare placeholders and create computation graph
    x = tf.placeholder(tf.float32, shape=[None, 10], name="input_characteristics")
    y_out = create_simple_network(x)
    y_ = tf.placeholder(tf.float32, shape=[None, 1000], name="known_books")

    ## Create loss calculation operation and train operation
    with tf.name_scope("train"):
        loss = loss_function(y_, y_out)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        op_summary_loss = tf.summary.scalar('loss', loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Initialise summary writer with graph used in the current session
        writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
        for epoch in range(num_epochs):
            train_step.run(feed_dict={x: train_chars, y_: train_books})

            ## Write tensorboard report data and console output
            if epoch % summary_frequency == 0:
                # Rather than running and adding all the summaries one at a time; merge, run and add in one go
                report_dict = {x: train_chars, y_: train_books}
                writer.add_summary(tf.summary.merge_all().eval(feed_dict=report_dict), epoch)
                # Output accuracy to console
                train_loss = loss.eval(feed_dict={x: train_chars, y_: train_books})
                test_loss = loss.eval(feed_dict={x: test_chars, y_: test_books})
                print('epoch ' + str(epoch) + ', loss on training data ' + str(train_loss) +
                      ', and on test data ' + str(test_loss))

        ## Evaluate network accuracy on the test data set
        print('Final loss on test data set: ' +
              str(loss.eval(feed_dict={x: test_chars, y_: test_books})))

        print('Saving trained model to ' + model_path)
        saver.save(sess, model_path)

        return y_out, writer


def recommend_with_network(model_path, user_chars):

    ## Declare placeholders and create computation graph
    x = tf.placeholder(tf.float32, shape=[None, 10], name="input_characteristics")
    y_out = create_simple_network(x)
    y_ = tf.placeholder(tf.float32, shape=[None, 1000], name="known_books")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        chars_matrix = user_chars.as_matrix()[:, 1:]
        results = sess.run(y_out, feed_dict={x:chars_matrix})
        books = list(map(com.top10, results))
        print("user, recommended book numbers")
        for idx, book in enumerate(books):
            print(user_chars.users[idx], book)


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
        train_network(args.epochs, train_books, test_books, train_chars, test_chars, model_path=args.model_file)

    elif args.recommend:
        print('Generating recommendations for users from ' + args.characteristics)
        user_chars = hlp.read_csv_user_chars(args.characteristics)
        recommend_with_network(args.model_file, user_chars)

if __name__ == "__main__":
    main()
