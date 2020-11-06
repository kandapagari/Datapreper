import os
import json
import random
import numpy as np

import cv2


def load_image(image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_one_hot(targets, nb_classes=2):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def resize_image_list(image_list, size=(100, 200)):
    result_images = []
    for image in image_list:
        result_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        result_images.append(result_image)
    return result_images


class TripletGenerator:
    def __init__(self, dataset_path, batch_size, seed=42):
        random.seed(seed)
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.patch_id_mapping = {}
        self.id_mapping = {}
        self.__init_patch_id()
        with open('backup.json', 'w') as j:
            json.dump({'Patch id mapping': self.patch_id_mapping, 'id list': self.id_mapping}, j)

    def __init_patch_id(self):
        for subfolder in os.listdir(self.dataset_path):
            if subfolder not in self.patch_id_mapping:
                self.patch_id_mapping[subfolder] = {}
            if subfolder not in self.id_mapping:
                self.id_mapping[subfolder] = []
            all_patch_names = os.listdir(os.path.join(self.dataset_path, subfolder))
            for file in all_patch_names:
                _id = file.split('_')[-1].replace('.jpg', '')
                self.id_mapping[subfolder].append(_id)
                if _id not in self.patch_id_mapping[subfolder]:
                    self.patch_id_mapping[subfolder][_id] = []
                self.patch_id_mapping[subfolder][_id].append(os.path.join(self.dataset_path, subfolder, file))
            self.id_mapping[subfolder] = list(set(self.id_mapping[subfolder]))
        print('Loading dataset complete..')

    def __call__(self, *args, **kwargs):
        for subfolder in os.listdir(self.dataset_path):
            for _id in self.id_mapping[subfolder]:
                while len(self.patch_id_mapping[subfolder][_id]) != 0:
                    batch_x = []
                    batch_y = []
                    for batch_mate in range(self.batch_size):
                        x, y = self.gen_triplet(subfolder, _id)
                        batch_x.append(x)
                        batch_y.append(y)
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    yield batch_x, batch_y

    def gen_triplet(self, subfolder, _id):
        first_image_name = self.get_random_item(_id, subfolder)
        second_image_name = self.get_random_item(_id, subfolder)
        random_id = random.choice([a for a in self.id_mapping[subfolder] if a is not _id])
        third_image_name = self.get_random_item(random_id, subfolder)
        first_image = load_image(first_image_name)
        second_image = load_image(second_image_name)
        third_image = load_image(third_image_name)
        image_list = [first_image, second_image, third_image]
        image_list = resize_image_list(image_list)
        label_list = [1, 1, 0]  # [_id, _id, random_id]
        temp = list(zip(image_list, label_list))
        random.shuffle(temp)
        image_list, label_list = zip(*temp)
        image_list, label_list = np.array(image_list), np.array(label_list)
        x_stack = np.stack(image_list, axis=0)
        y_stack = get_one_hot(label_list, 2)
        return x_stack, y_stack

    def get_random_item(self, _id, subfolder):
        name = random.choice(self.patch_id_mapping[subfolder][_id])
        self.patch_id_mapping[subfolder][_id].remove(name)
        return name


if __name__ == '__main__':
    trip_gen = TripletGenerator(r'C:\Users\785pa\Downloads\MOT16\train\output', 16)
    next(trip_gen())
    next(trip_gen())
