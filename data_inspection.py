import os
import cv2
import argparse
from tqdm import tqdm
import statistics


def get_mean_image_size(dataset_path):
    heights = []
    widths = []
    for subfolder in os.listdir(dataset_path):
        for file in tqdm(os.listdir(os.path.join(dataset_path, subfolder))):
            image = cv2.imread(os.path.join(dataset_path, subfolder, file))
            heights.append(image.shape[0])
            widths.append(image.shape[1])
    print(len(heights), len(widths))
    mean_height, mean_width = statistics.mean(heights), statistics.mean(widths)
    print(mean_height, mean_width)
    return mean_height, mean_width


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='path of dataset to patch', default=os.getcwd())

    args = parser.parse_args()
    mean_height, mean_width = get_mean_image_size(args.input)
