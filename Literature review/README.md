# Braille recognition models and performance results

This directory contains an overview of the different techniques utilised by different publications in the field of optical Braille recognition (OBR).

## Data

Each of the `tsv` files in the `data` subdirectory contain lists of publications associated with a particular technique, dataset, or focal point in research.  
These publications are labelled by an identifier in the form `<first_author><year_published>`.
For example, `Calders1986` identifies the seminal OBR paper _'Optical Pattern Recognition Of Braille Originals'_ by Calders, and Mennens and Francois, published in 1986.

The full citation details for each publication are included in BibTex format in `references.bib`, using the above identifiers for each.

Lastly, the `connected_papers.txt` file in the `data` subdirectory contains for each such publication, the list of further publications in the OBR field which cited the given paper (to the best of the authors' knowledge).

## Visualisations

These records were used in `visualisations.ipynb` to visualise varying aspects and techniques within the field, using the publication years to visualise trends over time, and using connected papers to emphasise impactful papers that directly contributed to future publications -- at least insofar these papers were cited within the field.  
Many of these visualisation were utilised in both the published journal article, and the Masters thesis submitted at SU.
