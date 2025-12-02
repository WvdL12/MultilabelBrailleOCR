"""
(c) Ilya Ovodov, 2023
https://github.com/IlyaOvodov/AngelinaDataset

Modified by Wicus van der Linden, 2023

Annotation demo: display image with annotation drawn on it
"""
from pathlib import Path
import PIL
from PIL import ImageDraw, ImageFont
import sys
import os

import numpy as np

from angelina import label_tools as lt
from angelina import data

# import label_tools as lt
# import data


def get_anno(fn, label_dict=None, sz=None):
    if fn.suffix == ".json":
        ann = data.read_LabelMe_annotation(fn)
        img = PIL.Image.open(fn.with_suffix(".jpg"))
    elif fn.suffix == ".txt":
        ann = data.read_yolo_annotation(fn, label_dict)
        img = PIL.Image.open(fn.with_suffix(".png"))
    else:
        ann = data.read_csv_annotation(fn.with_suffix(".csv"))
        img = PIL.Image.open(fn.with_suffix(".jpg"))
    draw = PIL.ImageDraw.Draw(img)
    if sz is None:
        base = 1 + (label_dict is not None)
        sz = int(10 + base * (np.mean(img.size) // 250))
    fnt = ImageFont.truetype("himalaya.ttf", sz)
    for left, top, right, bottom, label in ann:
        if fn.suffix != ".txt":
            label = lt.int_to_label123(label)
        draw.rectangle([left * img.width, top * img.height, right * img.width, bottom * img.height], outline='blue')
        draw.text((left * img.width, bottom * img.height),
                  label,
                  font=fnt,
                  fill="black")
    return img


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python annotation.py <image or annotation filename>")
        print("Displaying default sample.")
        filename = os.path.join('..', 'Data Raw', 'Angelina Dataset', 'uploaded', 'test2',
                                '0c62b07dc3ff44e4982cbb573f3b6d95.labeled.json')
    else:
        filename = sys.argv[1]
    image = get_anno(Path(filename))
    image.show()
