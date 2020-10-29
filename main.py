import argparse
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        raise cv2.Error
    return img[y1:y2, x1:x2, :]


def run_patcher(dataset_path, overlap):
    for subfolder in os.listdir(dataset_path):
        gt_path = os.path.join(dataset_path, subfolder, 'gt', 'gt.txt')
        gt = pd.read_csv(gt_path, delimiter=',', header=None,
                         names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'conf', 'z'])
        for i in range(1, gt.frame.max() + 1):
            frame_info = gt[gt.frame == i]
            image_path = os.path.join(dataset_path, subfolder, 'img1', f'{i:06}.jpg')
            image = cv2.imread(image_path)
            # plt.imshow(image)
            # plt.show()
            for _, info in frame_info.iterrows():
                bb_top, bb_left, bb_width, bb_height = info.bb_top, info.bb_left, info.bb_width, info.bb_height
                x1, y1, x2, y2 = bb_left, bb_top - bb_height, bb_left + bb_width, bb_top
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                pt1 = (bb_left, bb_top - bb_height)
                pt2 = (bb_left + bb_width, bb_top)
                color = (255, 0, 0)
                overlay = image.copy()
                cv2.rectangle(overlay, (x1, y1, x2, y2), color=color, thickness=3)
                plt.imshow(overlay)
                plt.show()
                centre_x, centre_y = (x1 + x2) // 2, (y1 + y2) // 2
                new_height = bb_height * (1 + overlap)
                new_width = bb_width * (1 + overlap)
                crop = imcrop(image, (centre_y - bb_width // 2, centre_x - bb_height // 2,
                                      centre_y + bb_width // 2, centre_x + bb_height // 2,))
                # plt.imshow(crop)
                # plt.show()
                bbox = (centre_y - new_width // 2, centre_x - new_height // 2,
                        centre_y + new_width // 2, centre_x + new_height // 2,)
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image, (x1, y1, x2, y2), color=(0, 255, 255), thickness=3)
                plt.imshow(image)
                plt.show()
                # plt.imshow(crop)
                # plt.show()
                print(bbox)
            print(image_path)

        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='path of dataset to patch', default=os.getcwd())
    parser.add_argument('-o', '--overlap', help='float of overlap for patches', default=0.2)

    args = parser.parse_args()
    train_path = os.path.join(args.input, 'train')
    run_patcher(train_path, args.overlap)
