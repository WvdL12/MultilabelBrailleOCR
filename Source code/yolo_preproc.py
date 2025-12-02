from sys import argv, exit
import glob, os, json
import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw
from math import ceil

DATA_RAW = os.path.join('..', 'Data Raw')
DATA_PROC = os.path.join('..', 'Datasets')
ANG = os.path.join('Angelina Dataset')
DSBI = os.path.join('DSBI')

def label_ang(folder, train, val, test):

    data_origin = "{}_{}".format(folder.split(os.sep)[-2], folder.split(os.sep)[-1])
    # print(data_origin)
    with open(os.path.join('dicts', "ang_dict.json"), encoding='utf8') as dict_file:
        ang_dict = json.load(dict_file)
    
    N = len(glob.glob(os.path.join(DATA_RAW, folder, "*.jpg")))
    test_N = ceil(0.1 * N)
    train_N = N - 2 * test_N
    
    if N == 1:
        indices = np.array([0])
    if N == 2:
        indices = np.array([0, 1])
        np.random.seed(1)
        if np.random.rand() > 0.5:
            indices -= 1
    else:
        indices = np.append(np.append(np.zeros(train_N), np.ones(test_N)), - np.ones(test_N))
    np.random.seed(101010)
    train_test_split = np.random.permutation(indices).astype(int)
    
    sample_count = 0
    obj_count = 0
    for filename in glob.glob(os.path.join(DATA_RAW, folder, "*.jpg")):
        # print(filename)
        label = filename.replace(".jpg", ".json")
        sample = cv2.imread(filename)
        subset_out = test if train_test_split[sample_count] > 0 else val if train_test_split[sample_count] < 0 else train

        label_dict = json.load(open(label))
        try:
            cv2.imwrite(os.path.join(subset_out, "images", "{}_{}.png".format(data_origin, sample_count)), sample)
        except cv2.error:
            print("Error on saving image file", filename)
        sample_h, sample_w, _ = sample.shape
        resize_dims = np.array([0.002 * sample_w, 0.002 * sample_h])

        with open(os.path.join(subset_out, "labels", "{}_{}.txt".format(data_origin, sample_count)), 'w') as label_file:
            for item in label_dict["shapes"]:
                coord0 = np.array(item["points"][0]) - resize_dims
                coord1 = np.array(item["points"][1]) + resize_dims
                coord_avg = 0.5 * coord0 + 0.5 * coord1
                item_w = (coord1[0] - coord0[0]) / sample_w
                item_h = (coord1[1] - coord0[1]) / sample_h
                # normalise
                coord_avg[0] = coord_avg[0] / sample_w
                coord_avg[1] = coord_avg[1] / sample_h
                
                try:
                    lbl = [int(a) for a in ang_dict[item["label"]]]
                except KeyError:
                    print("Key error in file {}".format(filename))
                    print("Key: {}; Coord0: {}".format(item["label"], coord0))
                    lbl = [0]*6
                
                labels = [i for i, l in enumerate(lbl) if l > 0]
                # print(f"Split label string {lbl}, label indices {labels}")

                lbl_str = f"{coord_avg[0]} {coord_avg[1]} {item_w} {item_h}"
                for l in labels:
                    label_file.write(f"{l} {lbl_str}\n")
                obj_count += 1
        sample_count += 1
    print("Processed {}, samples: {}, braille characters: {}".format(folder, sample_count, obj_count))


def label_dsbi(folder, train, val, test):

    data_origin = "{}_{}".format(folder.split(os.sep)[-2], folder.split(os.sep)[-1])
    # print(data_origin)
    
    img_list = [file for file in glob.glob(os.path.join(DATA_RAW, folder, "*.jpg")) if file.endswith("recto.jpg")]
    # print(f"Number of total pages: {len(img_list)}")
    for img in img_list:
        with open(img.replace('.jpg',  '.txt'), 'r') as label_file:
            if len(label_file.readlines()) == 0:
                # print(img)
                img_list.remove(img)
    # print(f"Number of pages after filtering verso-only: {len(img_list)}")
    
    N = len(img_list)
    sample_count = 0
    obj_count = 0
    for filename in img_list:
        label = filename.replace(".jpg", ".txt")
        sample = cv2.imread(filename)
        subset_out = test

        label_input = open(label, 'r').readlines()
        label_input.pop(0)
        col_coords = [int(l) for l in label_input.pop(0).split(" ")]
        row_coords = [int(l) for l in label_input.pop(0).split(" ")]
        # print(f"Sample {filename}: {len(row_coords)} rows, {len(col_coords)} columns.")
        sample_h, sample_w, _ = sample.shape
        resize_h, resize_w = 0.005 * sample_h, 0.01 * sample_w
        
        # draw_sample(sample, row_coords, col_coords, resize_w, resize_h, sample_w, sample_h)
        # break
        try:
            cv2.imwrite(os.path.join(subset_out, "images", "{}_{}.png".format(data_origin, sample_count)), sample)
        except cv2.error:
            print("Error on saving image file", filename)

        with open(os.path.join(subset_out, "labels", "{}_{}.txt".format(data_origin, sample_count)), 'w') as label_file:
            for str_item in label_input:
                item = [int(s) for s in str_item.split(" ")]
                r_start, c_start = (item[0] - 1) * 3, (item[1] - 1) * 2
                # print(f"Character: {item}, starting at row {r_start} column {c_start}")
                coord0 = np.array([row_coords[r_start] - resize_h, row_coords[r_start+2] + resize_h]) # y_coords
                coord1 = np.array([col_coords[c_start] - resize_w, col_coords[c_start+1] + resize_w]) # x_coords
                item_w = (coord1[1] - coord1[0]) / sample_w
                item_h = (coord0[1] - coord0[0]) / sample_h
                coord_avg = [sum(0.5 * coord1), sum(0.5 * coord0)]
                # normalise
                coord_avg[0] = coord_avg[0] / sample_w
                coord_avg[1] = coord_avg[1] / sample_h
                labels = [i for i, l in enumerate(item[2:]) if l > 0]
                # print(f"Split label string {item[2:]}, label indices {labels}")

                lbl_str = f"{coord_avg[0]} {coord_avg[1]} {item_w} {item_h}"
                for l in labels:
                    label_file.write(f"{l} {lbl_str}\n")
                obj_count += 1
        sample_count += 1
    print("Processed {}, samples: {}, braille characters: {}".format(folder, sample_count, obj_count))


def draw_sample(sample, row_coords, col_coords, resize_w, resize_h, sample_w, sample_h):
    img = Image.fromarray(sample)
    draw = ImageDraw.Draw(img)
    for r, row in enumerate(row_coords):
        if r % 3 == 1:
            continue
        op = -1 if r % 3 == 0 else 1
        draw.line([0, row + op * resize_h, sample_w, row + op * resize_h])
    for c, col in enumerate(col_coords):
        op = -1 if c % 2 == 0 else 1
        draw.line([col + op * resize_w, 0, col + op * resize_w, sample_h])
    img.show()
    

def main():
    if len(argv) < 4:
        print('Usage: python3 src/yolo_preproc.py <dataset: {"ang", "dsbi"}> <subset> <dataset_name>')
        exit()
    sub = argv[2]
    set_name = argv[3]
    
    out = os.path.join(DATA_PROC, 'braille_{}'.format(set_name))
    
    train_out = os.path.join(out, "train")
    val_out = os.path.join(out, "valid")
    test_out = os.path.join(out, "test")
    if not os.path.isdir(out):
        os.mkdir(out)
        os.mkdir(train_out)
        os.mkdir(os.path.join(train_out, "images"))
        os.mkdir(os.path.join(train_out, "labels"))
        os.mkdir(test_out)
        os.mkdir(os.path.join(test_out, "images"))
        os.mkdir(os.path.join(test_out, "labels"))
        os.mkdir(val_out)
        os.mkdir(os.path.join(val_out, "images"))
        os.mkdir(os.path.join(val_out, "labels"))
        
    if argv[1] == "ang":
        print("Processing {}: {}".format("Angelina", sub))
        sub_list = ["red", "cdr", "kov", "mddc", "mddr", "ola", "skaz", "tel", "t2", "up"]
        if sub == "all":    
            for folder in sub_list:
                label_ang(os.path.join(ANG, folder), train_out, val_out, test_out)
        else:
            label_ang(os.path.join(ANG, sub), train_out, val_out, test_out)
    elif argv[1] == "dsbi":
        print("Processing {}: {}".format(DSBI, sub))
        sub_list = ["FM", "Massage", "Math", "ODP", "SYF", "Book1", "Book2"]
        if sub == "all":    
            for folder in sub_list:
                label_dsbi(os.path.join(DSBI, folder), train_out, val_out, test_out)
        else:
            label_dsbi(os.path.join(DSBI, sub), train_out, val_out, test_out)
    else:
        print("Invalid argument option.")


if __name__ == "__main__":
    main()
