# Braille data, OBR models and performance results

## Datasets

Datasets available for use sourced from  

* [Ilya Ovodov, Angelina Set](https://github.com/IlyaOvodov/AngelinaDataset)  
* [DSBI Dataset](https://github.com/yeluo1994/DSBI)  

These datasets contain both Onesided and Twosided Braille documents and characters, with labels.  
The data labeling has been standardised from the different datasets. To account for generality, the label system used describes the dots present in a given Braille character, rather than directly translating the character.  
Processed datasets and records of train-test splits are published on Zenodo: `10.5281/zenodo.17453802`.  

## Models

The architecture space for CNN models was explored using the Iterated F-Race optimisation algorithm.
The logs of hyperparameter configurations explored, and subsequent loss performances, were recorded in JSON format, stored under the `Hyperparameter tuning` directory.  
The final model configurations, under different training sets and modelling approaches, were retrained, and model weights and configurations are stored under the `Saved Models` directory.  

## Performance results

The `Model Performance Results` directory includes performance results and model predictions used in masters research at Stellenbosch University, and included in an article accepted for publishing under the Journal of Universal Computer Science.
Models were trained under different data resampling scenarios, and with different classification approaches -- namely, multiclass and multilabel classification.
The predictions and performances on test sets include in-distribution test results on the Angelina dataset, which was used in training, as well as out-of-distribution (ood) test results on the DSBI dataset.
Experimental augmentations include Brightness, Noise, Rotation and Blur experiments to evaluate the robustness of each model to extreme data conditions.

## Source Code

Key extracts of the source code used in development of these models, as well as performance analysis and visualisation, is included in the `Source Code` directory.  
This includes the model utility libaries, performance metric formulae, data processing scripts, model training and performance evaluation scripts necessary to replicate the research, or apply to different subjects or applications.
