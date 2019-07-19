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

    imgarray = np.array([img_to_array(load_img(Path(args.testpath), target_size=(img_rows, img_cols)))])

    preds = model.predict(imgarray, batch_size=imgarray.shape[0])
    for pred in preds:
        predR = np.round(pred)
        for pre_i in np.arange(len(predR)):
            if predR[pre_i] == 1:
                print(f"num is '{pairs[str(pre_i)]}'")


if __name__ == '__main__':
    main()
