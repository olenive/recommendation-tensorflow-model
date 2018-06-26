import pandas as pd
import argparse
import tensorflow as tf


def interpret_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--characteristics', type=str, default="data/user_char.csv",
        help='Path to CSV file containing matrix of user characteristics.'
    )
    parser.add_argument(
        '-b', '--books', type=str, default="data/user_book.csv",
        help='Path to CSV file containing matrix of books liked by each user.'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='Random seed to be used for this run.'
    )
    parser.add_argument(
        '-s', '--test-set-length', type=int, default=0,
        help='Number of users to be used for the test set while training.'
    )
    parser.add_argument(
        '-e', '--total-epochs', type=int, default=20,
        help='Number of epochs for which the network will be traind.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-t', '--train', action='store_true',
        help='Train network using the indicated user_book and user_chars input files.'
    )
    group.add_argument(
        '-r', '--recommend', action='store_true',
        help='Make recommendations for users '
    )
    return parser.parse_args()


def read_csv_user_chars(path):
    df = pd.read_csv(path)
    df = df.drop(labels='Unnamed: 11', axis=1)
    df = df.rename(index=str, columns={'Unnamed: 0' : 'users'})
    return df


def read_csv_user_books(path):
    df = pd.read_csv(path)
    df = df.drop(labels='Unnamed: 1001', axis=1)
    df = df.rename(index=str, columns={'Unnamed: 0': 'users'})
    return df


def create_training_and_testing_sets(user_chars, user_books, test_set_length):
    # Note the following code assumes that the order of users in the input files matches.
    # Further work could involve randomly shuffling users.
    num_users = len(user_books)
    train_books = user_books[0: num_users - test_set_length]
    test_books = user_books[num_users - test_set_length:]
    train_chars = user_chars[0: len(train_books)]
    test_chars = user_chars[len(train_books): len(train_books) + len(test_books)]
    return train_books, test_books, train_chars, test_chars


def attach_summaries_(var, name="summaries"):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries/' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
