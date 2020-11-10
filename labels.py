import numpy as np


def label_creator(ids=None, nb_classes=2):
    if ids is None:
        ids = [1, 1, 0]
    if isinstance(ids, list):
        ids = np.array(ids)
    res = np.eye(nb_classes)[np.array(ids).reshape(-1)]
    return res.reshape(list(ids.shape) + [nb_classes])


print(label_creator())
