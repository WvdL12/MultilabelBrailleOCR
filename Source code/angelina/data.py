"""
(c) Ilya Ovodov, 2023
https://github.com/IlyaOvodov/AngelinaReader
https://github.com/IlyaOvodov/AngelinaDataset

Modified by Wicus van der Linden, 2023

Utilities to read braille annotation from LabelMe json and CSV files
Updated to include YOLO annotation
"""
import csv
import json5 as json

from angelina import label_tools as lt
# import label_tools as lt


def limiting_scaler(source, dest):
    """
    Creates function to convert coordinates from source scale to dest with limiting to [0..dest)
    :param source: source scale
    :param dest: dest scale
    :return: function f(x) for linear conversion [0..sousce)->[0..dest) so that
        f(0) = 0, f(source-1) = (source-1)/source*dest, f(x<0)=0, f(x>=source) = (source-1)/source*dest
    """

    def scale(x):
        return int(min(max(0, x), source - 1)) * dest / source

    return scale


def read_LabelMe_annotation(label_filename):
    """
    Reads LabelMe (see https://github.com/IlyaOvodov/labelme labelling tool) annotation JSON file.
    :param label_filename: path to LabelMe annotation JSON file
    :return: list of rect objects. Each rect object is a tuple (left, top, right, bottom, label, score) where
        left..bottom are in [0,1), label is int in [1..63]
        score is 1.0 if no 'score' key for the item. Score is set in auto-generated annotation
    """
    with open(label_filename, 'r', encoding='cp1251') as opened_json:
        loaded = json.load(opened_json)
    convert_x = limiting_scaler(loaded["imageWidth"], 1.0)
    convert_y = limiting_scaler(loaded["imageHeight"], 1.0)
    rects = [(convert_x(min(xvals)),
              convert_y(min(yvals)),
              convert_x(max(xvals)),
              convert_y(max(yvals)),
              lt.human_label_to_int(label),
              score
              ) for label, xvals, yvals, score in
             ((shape["label"],
               [coords[0] for coords in shape["points"]],
               [coords[1] for coords in shape["points"]],
               shape.get("score", 1.0)
               ) for shape in loaded["shapes"]
              )
             ]
    return rects


def read_csv_annotation(label_filename):
    """
    Reads CSV annotation with each line representing a single Braille char as:
    left;top;right;bottom;label
    :param label_filename: path to CSV annotation file
    :return: list of rect objects. Each rect object is a tuple (left, top, right, bottom, label) where
        left..bottom are in [0,1), label is int in [1..63]
    """
    rects = []
    with open(label_filename) as f:
        csv_reader = csv.reader(f, delimiter=';')
        for left, top, right, bottom, label in csv_reader:
            rects.append((float(left), float(top), float(right), float(bottom), int(label)))
    return rects


def read_yolo_annotation(label_filename, label_dict=None):
    """
    Reads TXT annotation with each line representing a single Braille char as:
    class center_x center_y width height label
    :param label_filename: path to TXT annotation file
    :return: list of rect objects. Each rect object is a tuple (left, top, right, bottom, label) where
        each is in [0,1) except label, which is a 6-bit binary stream with confidence score
    """
    rects = []
    if label_dict is not None:
        label_dict = json.load(open(label_dict, encoding='utf8'))
    with open(label_filename) as f:
        for line in f.readlines():
            splits = line.split(" ")
            center_x, center_y, width, height = [float(x) for x in splits[1:5]]
            label, label2 = splits[-2].split('_')
            if label_dict is not None:
                if '*' in label2:
                    j = label2.find('*')
                    label2 = label2[:j] + label2[j+1:]
                if label in label_dict:
                    label = label_dict[label]
                elif label2 in label_dict:
                    label = "*" + label_dict[label2]
                else:
                    label = "XX"
            width = 0.5 * width
            height = 0.5 * height
            left, right = center_x - width, center_x + width
            top, bottom = center_y - height, center_y + height
            # label = f"{label}\n{round(conf, 3) * 100}"
            rects.append((float(left), float(top), float(right), float(bottom), label))
    return rects
