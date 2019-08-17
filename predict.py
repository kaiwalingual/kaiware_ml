from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
from pathlib import Path

import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='mymodel.h5', required=True)
    parser.add_argument('--testpath', '-t', required=True)
    args = parser.parse_args()

    num_classes = 7
    img_rows, img_cols = 128, 128

    pairs = {}
    with open("pairs.json") as f:
        pairs = json.load(f)

    model = load_model(args.model)

    base_dir = Path(args.testpath)

    imgarray = []

    for f in base_dir.iterdir():
        img = load_img(f, target_size=(img_rows, img_cols))
        arr = img_to_array(img)
        imgarray.append(arr)

    imgarray = np.array(imgarray)

    preds = model.predict(imgarray, batch_size=imgarray.shape[0])

    avg = 0
    count = 0
    for pred in preds:
        predR = np.round(pred)
        for pre_i in np.arange(len(predR)):
            if predR[pre_i] == 1:
                # print(f"num is '{pairs[str(pre_i)]}'")
                avg += int(pairs[str(pre_i)])
                count += 1
    avg /= count
    print(round(avg))


if __name__ == '__main__':
    main()
