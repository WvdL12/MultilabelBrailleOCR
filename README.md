# Braille OBR models and performance results

NOTICE: This work relates to an in-progress Masters degree at Stellenbosch University, South Africa.
As such, no code, data or models are currently contained in this repository.
The corresponding Masters thesis is set to be submitted by December 2025, and following evaluation, the relevant code, data and models will be published to this repository by February 2026.

## Related publications and citations

The following publications or online resources relate to the research and work done on this project.

* The processed Braille character datasets, metadata, and performance results are published as a [dataset on Zenodo](https://doi.org/10.5281/zenodo.17453802).
* Parts of the results and analyses were accepted for publishing on by the [Journal of Universal Computer Science](https://lib.jucs.org/) (J.UCS).
* The full results and analysis will be submitted as a BSc Computer Science Masters thesis at [Stellenbosch University](https://www.su.ac.za/en) (SU).

You can cite this repository directly as:

```text
van der Linden, W. J., Grobler, T. L., & van Zijl, L. Multilabel optical character recognition of Braille text at SU (Version 1.0.0) [Computer software]. https://github.com/WvdL12/MultilabelBrailleOCR
```

or you can use the following Bibtex details:

```text
@software{van_der_Linden_Multilabel_optical_character,
author = {van der Linden, Wicus J and Grobler, Trienko L and van Zijl, Lynette},
license = {CC-BY-4.0},
title = {{Multilabel optical character recognition of Braille text at SU}},
url = {https://github.com/WvdL12/MultilabelBrailleOCR},
version = {1.0.0}
}
```

## Datasets used

The below public datasets were used to train and evaluate our models  

* [Ilya Ovodov, Angelina Set](https://github.com/IlyaOvodov/AngelinaDataset)  
* [DSBI Dataset](https://github.com/yeluo1994/DSBI)  

These datasets contain both Onesided and Twosided Braille documents and characters, with labels.  
The data labeling has been standardised from the different datasets. To account for generality, the label system used describes the dots present in a given Braille character, rather than directly translating the character.  
Processed datasets and records of train-test splits are published on Zenodo, as linked above.  

## Models

The architecture space for CNN models was explored using the Iterated F-Race optimisation algorithm.
The logs of hyperparameter configurations explored, and subsequent loss performances, were recorded in JSON format, stored under the `Hyperparameter tuning` directory.  
The final model configurations, under different training sets and modelling approaches, were retrained, and model weights and configurations are stored under the `Saved models` directory.  

## Performance results

The `Model performance Results` directory includes performance results and model predictions used in masters research at Stellenbosch University, and included in an article accepted for publishing under the Journal of Universal Computer Science.
The same results are published on Zenodo, as linked above.  

Models were trained under different data resampling scenarios, and with different classification approaches -- namely, multiclass and multilabel classification.
The predictions and performances on test sets include in-distribution test results on the Angelina dataset, which was used in training, as well as out-of-distribution (ood) test results on the DSBI dataset.
Experimental augmentations include Brightness, Noise, Rotation and Blur experiments to evaluate the robustness of each model to extreme data conditions.

## Source Code

Key extracts of the source code used in development of these models, as well as performance analysis and visualisation, is included in the `Source code` directory.  
This includes the model utility libaries, performance metric formulae, data processing scripts, model training and performance evaluation scripts necessary to replicate the research, or apply to different subjects or applications.

## Licensing

This source code and assets in this repository are licensed under the MIT License.
