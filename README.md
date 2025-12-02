# Braille recognition models and performance results

This work on optical Braille recognition (OBR) was completed as part of a Masters degree of Computer Science at Stellenbosch University, South Africa.
The corresponding thesis, titled _'Generalised, multilingual, optical Braille recognition models_, was submitted in December 2025.
Parts of the work was further approved for publishing as a journal article, titled _'A Robust Dot-focused Classification Approach to Convolutional Braille Recognition'_.

This repository contains code, data, analyses and models meant to supplement the thesis and corresponding publications.

## Datasets used

The below public datasets were used to train and evaluate our models  

* The [Angelina Set](https://github.com/IlyaOvodov/AngelinaDataset), by Ilya Ovodov  
* The [DSBI Dataset](https://github.com/yeluo1994/DSBI), by Li _et al._  

These datasets contain both ooesided and twosided (interpoint) Braille documents and characters, with labels.  
The data labeling has been standardised from the different datasets. To account for generality, the label system used describes the dots present in a given Braille character, rather than directly translating the character.  
Processed datasets and records of train-test splits are published on Zenodo, as linked above.  

## Literature review

A review on current and previous research on OBR was done as part of the thesis submitted at SU.
For an overview of the different techniques utilised by different publications, see the `Literature review` directory.

## Models

The architecture space for CNN models was explored using the Iterated F-Race optimisation algorithm.
The logs of hyperparameter configurations explored, and subsequent loss performances, were recorded in JSON format, stored under the `Hyperparameter tuning` directory.  
The final model configurations, under different training sets and modelling approaches, were retrained, and model weights and configurations are stored under the `Saved models` directory.  

## Performance results

The `Model performance Results` directory includes performance results and model predictions used in masters research at SU, and included in an article accepted for publishing under J.UCS.
The same set of results are included with the preprocessed datasets, published on Zenodo.  

Models were trained under different data resampling scenarios, and with different classification approaches -- namely, multiclass and multilabel classification.
The predictions and performances on test sets include in-distribution test results on the Angelina dataset, which was used in training, as well as out-of-distribution (ood) test results on the DSBI dataset.
Experimental augmentations include Brightness, Noise, Rotation and Blur experiments to evaluate the robustness of each model to extreme data conditions.

## Source Code

Key extracts of the source code used in development of these models, as well as performance analysis and visualisation, is included in the `Source code` directory.  
This includes the model utility libaries, performance metric formulae, data processing scripts, model training and performance evaluation scripts necessary to replicate the research, or apply to different subjects or applications.

## Related publications and citations

The following publications or online resources relate to the research and work done on this project.

* The processed Braille character datasets, metadata, and performance results are published as a [dataset on Zenodo](https://doi.org/10.5281/zenodo.17453802).
* Parts of the results and analyses were compiled into a journal article, accepted for publishing on by the [Journal of Universal Computer Science](https://lib.jucs.org/) (J.UCS).
* The full results and analysis was submitted as a BSc Computer Science Masters thesis at [Stellenbosch University](https://www.su.ac.za/en) (SU).

### Repository

You can cite this repository directly as:

```text
van der Linden, W. J., Grobler, T. L., & van Zijl, L. Multilabel optical character recognition of Braille text at SU (Version 1.0.0) [Computer software]. https://github.com/WvdL12/MultilabelBrailleOCR
```

or you can use the following citation details (in BibTex format):

```text
@software{van_der_Linden_Multilabel_optical_character,
author = {van der Linden, Wicus J and Grobler, Trienko L and van Zijl, Lynette},
license = {MIT},
title = {{Multilabel optical character recognition of Braille text at SU}},
url = {https://github.com/WvdL12/MultilabelBrailleOCR},
version = {1.0.0}
}
```

### Article

The (tentative) full citation for the corresponding journal article is

```text
van der Linden, W.J., Grobler, T.L. and van Zijl, L. (2026). A robust dot-focused classification approach to convolutional Braille recognition. Journal of Universal Computer Science, vol. 32, no. 4.
```

with BibTex formatted details

```text
@article{vanderLinden2026,
  author = {van der Linden, Wicus J and Grobler, Trienko L and van Zijl, Lynette},
  title = {A Robust Dot-focused Classification Approach to Convolutional {Braille} Recognition},
  journal = {Journal of Universal Computer Science},
  year = {2026},
  volume = {32},
  number = {4},
  submitted = {10 June 2025},
  accepted = {24 October 2025}
}
```

### Thesis

The (tentative) full citation for the corresponding Masters thesis is

```text
van der Linden, W.J., Grobler, T.L. and van Zijl, L. "Generalised, multilingual, optical Braille recognition models". Stellenbosch, South Africa: Stellenbosch University, 2025.
```

with BibTex formatted details

```text
@mastersthesis{vanderLinden2025,
  title = {Generalised, multilingual, optical Braille recognition models},
  author = {van der Linden, Wicus J and Grobler, Trienko L and van Zijl, Lynette},
  year = {2025},
  type = {Master's Thesis},
  institution = {Stellenbosch University},
  location = {Stellenbosch, South Africa},
}
```

## Licensing

This source code and assets in this repository are licensed under the MIT License.
