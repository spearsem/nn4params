import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import add, dot
from keras.layers.core import Lambda


class ThetaLoss(Layer):
    """
    A custom layer to append a loss term based on a coefficient vector.
    """

    def __init__(self, **kwargs):
        """Construct a ThetaLoss layer."""
        super(ThetaLoss, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        """
        Execute the loss layer and pass through the coefficient vector.
        """
        # The layer assumes three input tensors: the parameter vector,
        # the vector of the true target values for the current mini-batch,
        # and the original vector of true covariate inputs ("x") for the
        # mini-batch.
        params, y_true, x_true = inputs

        # Use the Keras API to manually express the calculation of the loss
        # function. This will result from sum_i [ (X_i*params - Y_i)^2 ],
        # using dot products to express the calculation, where `i` ranges
        # over the samples in the current mini-batch.
        residuals = add([
            Lambda(lambda z: -z)(y_true),
            dot([x_true, params], axes=-1)
        ])

        # Take the scalar-valued Tensor containing the sum of residuals across
        # the mini-batch, and append it to the global loss function.
        self.add_loss(
            K.sum(dot([residuals, residuals], axes=-1))
        )

        # The layer is just a pass-through, so after appending the loss we need
        # to the overall loss function, we don't do any transformations and only
        # give back the parameter vector exactly as it came in.
        return params

    def get_output_shape_for(self, input_shape):
        """Return the shape of the parameter vector (layer output)."""
        return input_shape[0]

    def compute_output_shape(self, input_shape):
        """Return the shape of the parameter vector (layer output)."""
        return input_shape[0]
