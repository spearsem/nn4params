import numpy as np


if __name__ == "__main__":

    num_samples = 100000  # Size of the overall data sample.
    num_covariates = 5    # Number of covariates not counting bias term.

    # True coefficient vector, including bias term at the end.
    theta_true = np.asarray([[3], [-1], [2], [1.5], [0.6], [2]])

    # Generate covariate data, with one extra dimension for the bias.
    x_data = np.random.randn(num_samples, num_covariates + 1)

    # Overwrite whatever was generated in the bias column with 1.
    x_data[:, num_covariates] = 1

    # Calculate what the true target values are. This could be modified to
    # add some noise as well.
    y_data = np.dot(x_data, theta_true)

    # Place the target values into the 0th column and the covariate data in
    # the remaining columns, then save to a local file.
    data = np.hstack((y_data, x_data))
    np.save("regression_data.npy", data)
