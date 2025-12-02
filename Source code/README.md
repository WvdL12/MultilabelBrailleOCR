# Braille recognition models and performance results

This directory contains extracts of the source code used to develop, train and analyse different models in the study.
This includes the model utility libaries, performance metric formulae, data processing scripts, model training and performance evaluation scripts necessary to replicate the research, or apply to different subjects or applications.

## Utility packages

The `angelina` package includes data annotation, labeling and processing tools taken from the source code available with the [Angelina dataset](https://github.com/IlyaOvodov/AngelinaDataset) [1]. The version included here includes some modifications compared to the version on the aforementioned Github page, but the credit for the majority of the functionality in these tools belong to the respective author.

The `model_utils` package includes the core utilities and tools developed for this study.

- `augment.py` contains functions for rotating or reflecting Braille character samples, and identifying the corresponding new label after augmentation. This module further includes the code used to resampling and optimise the training data for each resampling scenario, with respect to the the objective for each scenario.
- `braille_utils.py` include key methods to translate between different encodings of Braile labels, such as between decimal and binary encoding, or between decimal and one-hot encoding.
- `cbr_model.py` contains the model wrapper and model definition used to create CNN Braille recognition model instances, as well as training and evaluation scripts.
- `f_race.py` is used to run one instance of hyperparamter optimisation through an Iterated F-Race [2,3], for the given modelling approach and training set.
- `metrics.py` defines all metrics and measures used in this study, including standard measures like accuracy, recall and precision, and other statistical measures such as Shannon entropy and label correlations. These measures were defined and coded manually in order to have better control over the returned values and formats, to ensure the metrics extend correctly to the formats and encodings utilised in the study.
- `torch_utils.py` contains reusable methods for working with Pytorch datasets and models.

## Data processing and analysis

This directory further contains python scripts and Jupyter notebooks used to process and analyse the different datasets used in this study.
This includes the aforementioned Angelina dataset [1] containing Russian Braille documents, as well as the [DSBI dataset](https://github.com/yeluo1994/DSBI) [4] of Chinese Mandarin Braille documents.

- `yolo_preproc.py` is a preprocessing script for processing through each document in the abovementioned datasets, standardising the label encoding to binary formats, and assigning each document to a train, validation or testing set based on a stratified 80-10-10 split. The name `yolo` alludes to the YOLO object detection models that were initially included in the scope of study, which informed the standardisation of the labels for Braille characters (objects) contained in each document image.
- `label_data.py` processes the individual documents for each of the train, validation and test sets produced above, and extracts individual Braille characters utilising the provided bounding box coordinates. These characters along with their binary encoded labels are standardised and saved as Numpy array datasets.
- `data_preprocessing.ipynb` contains initial explorations and analyses of the above datasets, as well as some supplementary datasets that were not used in the final study.
- `datasets.ipynb` includes more thorough analyses of the different datasets, including exploring the physical attributes and similarities, as well as the distribution of Braille character classes, dot labels, correlations, and so forth.

## Hyperparameter tuning

This pertains to the python scripts and Jupyter notebooks used to explore hyperparameter optimisation frameworks, and optimise the model configurations for different scenarios.

- `NAS_trials.ipynb` includes a preliminary exploration of both Bayesian optimisation [5] and the Iterated F-Race [2,3]. It was based on this exploration that the decision was made to use the Iterated F-Race as hyperparameter optimisation (or _neural architecture search_) algorithm.
- The optimisation process for each of the four resampling scenarios (see below) is contained in the pairs of files `<scenario_name>_CBR.ipynb` and `<scenario_abbreviation>_cbr_tuning.py`.
  - `<scenario_name>_CBR.ipynb` contains initial exploration of resampling strategies that informed the final resampling methods included in `model_utils/augment.py`, as well as a brief exploration of the final optimisation (or tuning) results, and the retraining of the optimised model candidate.
  - The `<scenario_abbreviation>_cbr_tuning.py` script was used to run the optimisation process for a given resampling scenario, for both multiclass and multilabel modelling approaches in turn, using the `model_utils/f_race.py` script and recording the results.
- `tuning_investigation_CBR.ipynb` includes more thorough analyses of the different hyperparameter optimisation results, including exploring the distribution of hyperparameter configurations explored, as well as the sensitivity of model performance to each hyperparameter, in each optimisation process.

## Model analysis

The final set of scripts and notebooks contain the different analyses and experiments conducted on the final optimised and trained models.

- `model_comparison_CBR.ipynb` includes a variety of model visualisation and exploration techniques employed to visualise different aspects of the final tuned networks, including saliency maps, concept vectors and feature map visualisations.
- `CBR_experimental_analysis.ipynb` contains an analysis of model performance over different experimental adverse conditions, obtained through augmenting the test set with changes to brightness, noise, rotation and blur.
- `model_performance.ipynb` contains the full analyses of model performances, including in-distribution, out-of-distribution, and experimental evaluations. Visual and statistical analyses are done to compare modelling approaches, different resampling scenarios, performance on minority vs majority classes, and performance at different levels of augmentation intensity. The recorded results are saved in csv formats in the `Model performance results` directory.

## References

[1] Ovodov, I.G. (2021a). Optical Braille recognition using object detection CNN. In: Pro￾ceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1741–1748. IEEE Computer Society, Los Alamitos, CA, USA.
[2] Birattari, M., Yuan, Z., Balaprakash, P. and Stützle, T. (2010). F-Race and Iterated F-Race: An Overview, pp. 311–336. Springer, Berlin, Heidelberg.
[3] Klazar, R. and Engelbrecht, A.P. (2014). Parameter optimization by means of statistical quality guides in F-Race. In: Proceedings of the IEEE Congress on Evolutionary Computation, pp. 2547–2552.
[4] Li, R., Liu, H., Wang, X. and Qian, Y. (2018). DSBI: double-sided Braille image dataset and algorithm evaluation for Braille dots detection. In: Proceedings of the 2nd International Conference on Video and Image Processing, ICVIP ’18, pp. 65–69.  ssociation for Computing Machinery, New York, USA.
[5] Mockus, J. (1998). The application of bayesian methods for seeking the extremum. Towards global optimization, vol. 2, pp. 117–129
