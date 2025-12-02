# Braille recognition models and performance results

This directory contains saved weights and attributes of the models developed in the study.

## Model training

The best performing model configurations for each of the eight optimisation conditions (see `Hyperparameter tuning` directory), were retrained on the full corresponding training set, and model weights and configurations are stored.

The training and validation loss, as well as training and validation accuracy, was recorded at each epoch and saved to the `history.json` file.
This file further contains the total number of training epochs, and the "convergence epoch" at which the validation loss was minimal.

The set of model weights at the convergence epoch is saved in the `model_weights.pt` file.

## Hyperparameters

Model parameters and hyperparameters are included in a `parameters.json` file for each trained model.

Parameters refer to simple arguments passed to the model constructor to properly initialise the class instance. This includes `in_size`, which dictates the dimensions of valid inputs to the input layer of the networkm and similarly `out_size` which dictates the number of output neurons (64 for multiclass, and 6 for multilabel models).
The remaining items in the file correspond to the optimised hyperparameter configuration:

- `lr`: Learning rate
- `beta_1`: $\beta_1$ (Adam optimiser)
- `beta_2`: $\beta_2$ (Adam optimiser)
- `l2`: $\ell^2$ regularisation penalty
- `filts`: Number of filters in first convolutional layer
- `kerns`: Kernel size
- `dense_sz`: Number of neurons in penultimate fully connected (dense) layer
- `activ`: Activation function used (ReLU, $\tanh$, LeakyReLU)
- `pad`: Binary indicator for Valid vs Same padding
- `grey_scaled`: Binary indicator for a grayscaling preprocessing step
- `square_in`: Binary indicator for a square aspect ratio preprocessing step

## Unique models

Independent optimisations were executed for eight distinct conditions, corresponding to two modelling approaches and four resampling scenarios.
Each model is labeled in the format `<resampling>_<modelling>_model[_version]`.
Multiclass models are identified with the `mc` modelling tag, while multilabel models are identified by `ml`.

The four resampling scenarios each utilised a different training dataset, with a different data resampling strategy applied.

- The `base` scenario corresponds to the training set with no resampling applied.
- The `cb`, or class balanced scenario, corresponds to resampling applied to minimise imbalance between Braille classes.
- The `lb`, or label balanced scenario, similarly applied resampling to minimise correlations between Braille dot labels.
- The `ab`, or adaptive balanced scenario, combined the above two objectives into a multi-objective resampling strategy.

The version tag optionally indicates a repeated optimisation process. This is only included in the `base` scenario, where version 1 corresponded to an initial test of the optimisation process, which led to refinements and minor changes that applied to version 2 of the `base` scenario, and all further scenarios.
