from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


def load(path: str, size=(100, 100), train_ratio=(9, 1)) -> Tuple[
    Tuple[np.array, np.array], Tuple[np.array, np.array], Dict[int, str]]:
    p = Path(path)

    if not p.exists() and p.is_dir():
        raise Exception("Path not found")

    loaded_data = {}

    for d in p.iterdir():
        if d.is_dir():
            loaded_data[d.name] = []

            for f in d.iterdir():
                if f.is_file():
                    img = load_img(f, target_size=size)
                    arr = img_to_array(img)
                    loaded_data[d.name].append(arr)

    X = []
    Y = []
    pairs = {}

    for idx, (k, v) in enumerate(loaded_data.items()):
        pairs[idx] = k
        for data in v:
            X.append(data)
            Y.append(idx)

    rate = int(len(X) * train_ratio[0] / (train_ratio[0] + train_ratio[1]))
    X_train = np.array(X[:rate])
    Y_train = np.array(Y[:rate])
    X_test = np.array(X[rate:])
    Y_test = np.array(Y[rate:])

    return (X_train, Y_train), (X_test, Y_test), pairs


if __name__ == "__main__":
    print(load("dataset", size=(20, 20), train_ratio=(2, 1)))
