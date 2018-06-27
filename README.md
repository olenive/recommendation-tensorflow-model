I initailly tried a fully connected neural network model with sigmoidal activation with just two layers, an input layer of length 10 and an output layer of length 1000.
This model did not converge well.  Adding more layers improved the result slightly but I think the amount of data is the main limiting factor.
Training the model past about 4000 epochs leads to overfitting (could try dropout) and poorer performance on the test set.

To train the model on 80 users with 10 users reserved as the test set, run the following command:

`python3 main.py --train --epochs 2000 --test-set-length 10`

To train the model on all 90 available users:

`python3 main.py --train --epochs 2000`

Run tensorboard (e.g. `tensorboard --logdir=tensorboard_output/`) to view the model graph and parameter summary statistics.

To obtain recommendations for selected users:

`python3 main.py --recommend --characteristics "data/test_user_char.csv"`

For further descriptions of command line arguments type `python main.py --help`.

