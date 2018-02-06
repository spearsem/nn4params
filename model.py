import keras.backend as K
import numpy as np
from keras.engine.topology import Input
from keras.layers.core import Activation, Dense
from keras.models import Model
from keras.optimizers import Adam

from theta_layer import ThetaLoss


# Parameters for model training
BATCH_SIZE = 512              # for simple regression data, larger batch sizes
                              # are better.

NB_EPOCH = 700                # this is an overkill number of training epochs,
                              # but it's useful to watch how the loss behaves
                              # during long training.

OPTIMIZER = Adam(decay=1e-7)  # because the model has many parameters for such
                              # a simple task, decaying the learning rate is
                              # helpful to avoid increasing training error over
                              # time.

TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
VERBOSE = True

# Load training data and store information about what the size and shape of the
# parameter vector will need to be.
INPUT_DATA = np.load("regression_data.npy")
INPUT_SIZE = INPUT_DATA.shape[1] - 1  # Number of covariates.
INPUT_SHAPE = (INPUT_SIZE, )


def split_data(data, test_fraction):
    """
    Splits data into (x_train, y_train), (x_test, y_test).

    By convention, it is assumed that the data already includes a bias term in
    the final column, and includes the target variable in the 0th column.
    """
    n = data.shape[0]
    num_train = int((1 - test_fraction) * n)
    d_train = data[:num_train, :]
    d_test = data[num_train:, :]
    y_train, x_train = d_train[:, 0], d_train[:, 1:]
    y_test, x_test = d_test[:, 0], d_test[:, 1:]
    return (x_train, y_train), (x_test, y_test)


def zero_loss(y_true, y_pred):
    """
    A custom loss function that ignores the final output of the network.

    This is used when we have otherwise defined or augmented the network's loss
    with a different loss function, and so by the time we reach the final
    output of the network, we do not need to compare it with the target
    associated variable.

    Note that for our model at hand, y_true would be a n-by-1 column vector
    where each scalar component is the target of the regression for that
    particular training sample. But since we are predicting a whole coefficient
    vector, y_pred will be an N-by-INPUT_SIZE matrix, each row containing the
    predicted regression coefficients based on that row's sample of data.

    To produce a loss by comparing these, we would also need the original
    covariate matrix X, and to look at sum [ (y_true - X*y_pred)^{2} ]. But by
    a limitation of the Keras API, we cannot access the covariate matrix X here
    in this loss function. So instead, we'll offload it to another layer to
    compute the loss, and in the final loss layer, we'll pretend like the loss
    is always zero.
    """
    return K.zeros_like(y_true)


if __name__ == "__main__":

    # Provide the training and testing data.
    (x_train, y_train), (x_test, y_test) = split_data(
        INPUT_DATA,
        TEST_SPLIT
    )

    # Define a network that will predict regression _coefficients_ from an
    # input. Note that because we will need a custom loss that penalizes a set
    # of coefficients based on how well they reproduce the training target
    # variables from the training covariates, the network is going to need to
    # explicitly consider the true target variables as an input.
    x_true_input = Input(shape=INPUT_SHAPE)
    y_true_input = Input(shape=(1, ))

    # Proceed with standard MLP layers based upon the input covariates. The
    # final layer of this section of the network should produce a coefficient
    # vector, so it must be the same size as the input data, and will not have
    # an activation.
    m = Dense(128)(x_true_input)
    m = Activation('relu')(m)
    m = Dense(256)(m)
    m = Activation('relu')(m)
    m = Dense(512)(m)
    m = Activation('relu')(m)
    m = Dense(INPUT_SIZE)(m)

    # Here is the trick! We're going to use our own custom layer that is
    # essentially a "pass through" layer. It will accept the vector of
    # predicted coefficients, along with the training data. Inside of this
    # layer, a contribution to the loss function will be calculated by
    # computing (Y - XB)^{2} on the tensors with the target variable, the
    # input data, and the current predicted coefficients. Then the layer
    # will just pass through the parameters.
    params = ThetaLoss()([m, y_true_input, x_true_input])

    # Define a model that accepts two tensor inputs, one of the covariates
    # and one for their associated target values, and has one tensor of
    # outputs, the predicted coefficients. Note that after we are done
    # training the model, we could input all zeros or gibberish or anything
    # we want for the target variable (`y_true_input`), since this will be
    # unknown whenever we encounter a new sample of the coavriate vector,
    # and we will not care about accumulate the loss function when we're not
    # performing model training.
    model = Model(
        inputs=[x_true_input, y_true_input],
        outputs=params
    )

    # Compile the model with the special zero_loss function. Because the
    # interface of Keras loss functions at the final layer must look at the
    # predicted value and the ground truth only (e.g. it cannot be made to
    # accept the original input covariate vector), it's of no use to us. The
    # target variable is a scalar whereas the model predicts a vector of
    # coefficients. We can't compute a loss at this stage unless we can take
    # the inner product of the coefficients with the covariate vector, to get
    # a scalar that should be compared to the target. That's why we already
    # perform this exact calculation in the ThetaLoss layer, and ensure it is
    # added to the global loss function.
    model.compile(loss=zero_loss, optimizer=OPTIMIZER)


    # Now we just fit the model with the training parameters, print the true
    # cofficients (assuming you use the provided `regression_data.npy`), and
    # print the average of the network's predictions across the test data.
    history = model.fit(
        [x_train, y_train],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=NB_EPOCH,
        validation_split=VALIDATION_SPLIT,
        verbose=VERBOSE
    )
    print(np.asarray([3, -1, 2, 1.5, 0.6, 2]))
    print(model.predict([x_test, y_test]).mean(axis=0))
