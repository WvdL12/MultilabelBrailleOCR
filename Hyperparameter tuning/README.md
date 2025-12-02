# Braille recognition models and performance results

This directory contains the logs of hyperparameter configurations explored, and subsequent loss performances, were recorded in JSON format.

## Hyperparameter optimisation

Hyperparameter optimisation was done through an Iterated F-Race [1,2] on a cross-validation split of a training set.
Numerous configurations were explored and iteratively refined to narrow down the search to an optimal candidate.

Each optimisation process yielded a set of elite candidates, which includes up to five similarly performance configurations, ranked by mean validation loss.
Furthermore, the full history of configurations explored and evaluated is recorded along with validation loss scores.

## Hyperparameters

Each model configuration includes 11 hyperparameters, listed in order below.

- Learning rate
- $\beta_1$ (Adam optimiser)
- $\beta_2$ (Adam optimiser)
- $\ell^2$ regularisation penalty
- Number of filters in first convolutional layer
- Kernel size
- Number of neurons in penultimate fully connected (dense) layer
- Categorical indicator for activation function used (ReLU, $\tanh$, LeakyReLU)
- Binary indicator for Valid vs Same padding
- Binary indicator for a grayscaling preprocessing step
- Binary indicator for a square aspect ratio preprocessing step

## Optimisation conditions

Independent optimisations were executed for eight distinct conditions, corresponding to two modeling approaches and four resampling scenarios.

The optimisation results for multiclass models are contained under the `mc_tuning` subdirectory, while the corresponding results for multilabel models are contained under `ml_tuning`.

The four resampling scenarios each utilised a different training dataset, with a different data resampling strategy applied.

- The `base` scenario corresponds to the training set with no resampling applied.
- The `cb`, or class balanced scenario, corresponds to resampling applied to minimise imbalance between Braille classes.
- The `lb`, or label balanced scenario, similarly applied resampling to minimise correlations between Braille dot labels.
- The `ab`, or adaptive balanced scenario, combined the above two objectives into a multi-objective resampling strategy.

## References

[1] Birattari, M., Yuan, Z., Balaprakash, P. and Stützle, T. (2010). F-Race and Iterated F-Race: An Overview, pp. 311–336. Springer, Berlin, Heidelberg.
[2] Klazar, R. and Engelbrecht, A.P. (2014). Parameter optimization by means of statistical quality guides in F-Race. In: Proceedings of the IEEE Congress on Evolutionary Computation, pp. 2547–2552.
