# nn4params
Learn to predict regression coefficients from data.

In section 6.2.2.4 of Deep Learning by Goodfellow, Bengio, and Courville,
it is mentioned that deep feedforward networks can be used to produce different
output types. Typically one thinks of a machine learning model as always
producing an output type that matches the target value of the learning task. In
a classification task, the model output would be a distribution among labels.
In a regression task, the model output would be a predicted value of a 
continuous target variable.

In that case, to evaluate the mechanism of the model, one would extract
parameters from it. As a simple example, consider a regression model

```
z ~ ax + by + c + eps
```

Given some training data containing `z` values (the target) and an n-by-2
matrix of data values (columns of `x` and `y`), you could fit many types
of models to predict `z` from a new set of `(x, y)`, including a linear
prediction described above.

The loss function of this set-up might often be a function of squared error.
Let `B` represent the set of coefficients to learn, `(a, b, c)`. And let
`X` represent the set of data for one sample, `(x, y, 1)`. Then the loss as a 
function of `B` is:


```
L(B) = (z - XB)^{2}
```

and it would lead to a model that can produce `z` values when given `x` and `y`
values.

Going back to the earlier statement: what if we want a model that accepts the
input data, `x` and `y`, and outputs *the parameter vector* `B`. Take a moment
to think about this. The model accepts the same kind of input data, but instead
of directly predicting a `z` value, it predicts the set of coefficients.

More generally, we might assume there is a parametric distribution generating
the target value (such as in the case of homoscedastic linear regression, where
the distribution is a Gaussian with mean given by the linear transformation of
the input values). This will lead directly to a loss function based on the
likelihood (or log likelihood) of the parameters given the training data.

In a standard setting, we might use direct optimization techniques to solve it.
We might use advanced multi-variate extensions of the Newton-Raphson method.
Or, as in the case with modern neural network frameworks, we might use
stochastic gradient descent. Optimizing the loss function would result in the
*parameters* of the underlying distribution.

Normally when training a neural network, there is some "runtime task" of
interest. That is, the reason for training the model is not for purposes of
looking at the parameters that are learned (except possibly for diagnostics or
debugging). Rather, the purpose is so that in some real "runtime" scenario,
such as a self-driving car experiencing some incoming image data, the model can
make a prediction directly in the target variable space which is actionable
in some way that matters to someone.

But if we set up our loss function to optimize for a set of distributional
parameters, we might instead say that the entire training set is solely for
the purpose of learning the parameters. After that, we will use the conditional
probability (with our learned parameters) if we want to guess what will
happen in a runtime scenario. Otherwise, we'll be content to have learned
what some parameters are.

This repository is meant to give an illustrative example of the preceding
paragraph. It uses the Keras library on top of TensorFlow and demonstrates
the concept of changing the network's loss function to reflect how well it
is learning a set of parameters, rather than how well it is *using* parameters
to make accurate guesses of a target variable.

The file `generate_data.py` can be used to generate a set of training data. A
pre-generated set is provided in `regression_data.npy`.

The model definition, with explanatory comment sections is found in 
`model.py`, which (assuming the requirements are satisfied) can be run with
`python model.py` (and will assume it is using `regression_data.npy`).

At the end of model training, the model is executed in forward mode across the
test split of the data set. For each example, the model outputs *the 
parameters* it believes a linear model requires to satisfy the distributional
assumption. By averaging across the test set, we arrive as an overall estimate
of the model parameters (we could also do this across the training set too).

To illustrate this, consider the output after the model has trained on
`regression_data.npy`, with the true coefficients printed first:

```
Epoch 697/700
81000/81000 [==============================] - 5s - loss: 0.0339 - val_loss: 0.0023
Epoch 698/700
81000/81000 [==============================] - 5s - loss: 3.4127e-04 - val_loss: 6.1571e-04
Epoch 699/700
81000/81000 [==============================] - 5s - loss: 9.1990e-05 - val_loss: 2.0417e-04
Epoch 700/700
81000/81000 [==============================] - 5s - loss: 0.0016 - val_loss: 0.0325

# true coefficients
[ 3.  -1.   2.   1.5  0.6  2. ]

# averaged output of the network across test set
[ 2.99166465 -0.9965589   1.99577916  1.49876416  0.59915525  1.99638963]
```

Keep in mind this is not a practical suggestion! In the case of linear
regression, it is much more computationally efficient to solve with a direct 
method like Gaussian elimination and matrix decomposition. Solving the problem
with neural networks, compelling the network to learn a non-linear
transformation of the data that produces the coefficients for a linear
transformation of the data, just adds extra work for the optimizer.

The point, rather, is to demonstrate the modeling power of neural networks. By
using a loss function that penalizes the networks ability to *predict the
parameters of a separate distribution*, the network can in fact estimate those
parameters directly.
