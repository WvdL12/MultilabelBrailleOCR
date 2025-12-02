from sys import argv, exit
import glob, os, json
import time
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from model_utils import bin_to_num, num_to_bin_string

SCRATCH = os.path.join('scratch')
DATA_IN = os.path.join('..', 'Datasets')
DATA_OUT = os.path.join('..', 'Datasets', 'numpy_datasets')
SUBS = ['train', 'valid', 'test']
# SUBS = ['train']

BRAILLE_W, BRAILLE_H = 30, 40


def segment_yolo(folder, out_name):
    counts = {}
    datasets = {}
    total = 0
    
    print('*'*100)
    print(f'Processing folder {folder}, with output {out_name}')
    
    for sub_folder in SUBS:
        print('*'*100)
        print(f'Sub-folder {sub_folder}')
        t1 = time.time()
        
        page_count, sample_count = 0, 0
        data_X, data_Y = [], []
        
        ##### DEBUG INFO
        sample_dims = [[], []]
        min_h, min_w = 100, 100
        #####
        
        for img_file in glob.glob(os.path.join(DATA_IN, folder, sub_folder, "images", "*.png")):
            lbl_file = img_file.replace('images', 'labels').replace('.png', '.txt')
            page_sample = cv2.imread(img_file)
            page_h, page_w, _ = page_sample.shape
            page_dims = np.array([page_w, page_h])
            
            labels_df = pd.DataFrame(columns=['Label', 'X', 'Y', 'W', 'H'])
            with open(lbl_file, 'r') as temp_file:
                for line in temp_file.readlines():
                    labels_df.loc[len(labels_df)] = [float(l) for l in line.split(" ")]
            
            for group, sample in labels_df.groupby(["X", "Y", "W", "H"]):
                center = np.array(group[0:2]) * page_dims
                dims = 0.5 * (np.array(group[2:4]) * page_dims)
                indices = list(sample['Label'].astype(int))

                coord0 = np.floor(center - dims).astype(int)
                coord1 = np.ceil(center + dims).astype(int)
                
                coord0[coord0 < 0] = 0
                coord1[coord1 < 0] = 0

                new_sample = page_sample[coord0[1]:coord1[1], coord0[0]:coord1[0]][:,:,[2,1,0]] # Record and transform to RGB Convention
                lbl = np.array([0]*6)
                lbl[indices] = 1

                ##### DEBUGGING INFO
                h, w, _ = new_sample.shape
                sample_dims[0].append(h)
                sample_dims[1].append(w)
                if h < min_h:
                    min_h = h
                if w < min_w:
                    min_w = w
                #####

                data_Y.append(lbl)
                # cv2.imwrite(os.path.join(DATA_OUT, "temp", f"sample_{sample_count}.png"), new_sample)
                try:
                    resized = cv2.resize(new_sample, (BRAILLE_W, BRAILLE_H), interpolation = cv2.INTER_AREA)
                    data_X.append(resized)
                except cv2.error:
                    print(f"Broken sample:  size {new_sample.shape}, filename {img_file}")
                    print(f"Original info {group}, processed dims {coord0} to {coord1}")

                sample_count += 1
            page_count += 1

        print(f"Processed {page_count} {sub_folder} pages in {round(time.time() - t1, 2)}s, total of {sample_count} characters.")
        counts[sub_folder] = sample_count
        datasets[sub_folder] = (np.array(data_X, dtype=np.int16), np.array(data_Y))
        total += sample_count
    
        ##### DEBUG INFO
        # print(f"Average sample dimensions {sample_dims[0] / total} x {sample_dims[1] / total}")
        mh = np.mean(sample_dims[0])
        mw = np.mean(sample_dims[1])
        print(f"Total samples seen: {len(sample_dims[0])} (according to indiv counting - {total})")
        print(f"Average sample dimensions {mh} x {mw}")
        print(f"Minimum height seen {min_h}, min width {min_w}")
        
        sns.scatterplot(x=sample_dims[1], y=sample_dims[0], label='Distribution', alpha=0.5)
        sns.kdeplot(x=sample_dims[1], y=sample_dims[0], fill=False, cut=0)
        sns.scatterplot(x=[mw], y=[mh], marker='x', label='Mean dimension', s=200)
        
        plt.title(f"Distribution of sample dimensions\nMean H: {round(mh, 3)}, W: {round(mw, 3)}")
        plt.xlabel('Width')
        plt.ylabel('Height')
        
        if not np.isnan(mw):
            plt.savefig(os.path.join(SCRATCH, f'{folder}_{sub_folder}_sample_dimension_distribution.png'), bbox_inches='tight')
        plt.cla()
        #####
        
    print(f"Processed {folder} dataset, sample counts:")
    for c in counts:
        print(f">>> {c}: {counts[c]} / {total} --- {round(100 * counts[c] / total, 3)}%")

    output = os.path.join(DATA_OUT, f"{out_name}.npz")
    train_X, train_Y = datasets['train']
    
    print(f"Training set dimesions --- X: {train_X.shape}, Y: {train_Y.shape}")
    test_X, test_Y = datasets['test'] if 'test' in datasets else (None, None)
    valid_X, valid_Y = datasets['valid'] if 'valid' in datasets else (None, None)
    np.savez(output, train_x=train_X, train_y=train_Y,
        val_x=valid_X, val_y=valid_Y, test_x=test_X, test_y=test_Y)
    if 'valid' in datasets:
        print(f"Validation set dimesions --- X: {valid_X.shape}, Y: {valid_Y.shape}")
    if 'test' in datasets:
        print(f"Test set dimesions --- X: {test_X.shape}, Y: {test_Y.shape}")


# def segment_kaggle():
#     with open(os.path.join(DICTS, "kag_dict.json")) as dict_file:
#         kag_dict = json.load(dict_file)
#         for filename in glob.glob(os.path.join(KAG, "*.jpg")):
#             sample = cv2.imread(filename)
#             name = filename[filename.rindex("/") + 1:]
#             lbl = kag_dict[name[0]]
#             out = os.path.join(DATA_PROC, "kaggle_os")
#             with open(os.path.join(out, "labels", name.replace(".jpg", ".txt")), 'w') as out_lbl:
#                 out_lbl.write(bin_to_int(lbl))
#                 out_lbl.close()
#             cv2.imwrite(os.path.join(out, "data", name), sample)


def main():
    if not len(argv) == 3:
        print('Usage: python3 src/label_data.py <folder_name output_name>')
        exit()
    out_name = argv[2]
    folder = argv[1]
    
    segment_yolo(folder, out_name)


if __name__ == "__main__":
    main()
