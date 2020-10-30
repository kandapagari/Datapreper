import argparse
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def imcrop(img, bbox):
    (x1, y1), (x2, y2) = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        (x1, y1), (x2, y2) = check_border((x1, y1), (x2, y2), img)
    (x1, y1), (x2, y2) = (x1, y2), (x2, y1)
    return img[y1:y2, x1:x2, :]


def check_border(pt1, pt2, img):
    max_height, max_width = img.shape[0], img.shape[1]
    pt1, pt2 = list(pt1), list(pt2)
    if pt1[0] > max_width:
        pt1[0] = max_width
    if pt2[0] > max_width:
        pt2[0] = max_width
    if pt1[1] > max_height:
        pt1[1] = max_height
    if pt1[1] > max_height:
        pt1[1] = max_height
    if pt1[0] < 0:
        pt1[0] = 0
    if pt2[0] < 0:
        pt2[0] = 0
    if pt1[1] < 0:
        pt1[1] = 0
    if pt2[1] < 0:
        pt2[1] = 0
    return tuple(pt1), tuple(pt2)


def run_patcher(dataset_path, overlap, output_root):
    if output_root is None:
        output_root = os.path.join(dataset_path, 'Output')
    if not os.path.isdir(output_root):
        os.makedirs(output_root)
    for subfolder in os.listdir(dataset_path):
        if os.path.join(dataset_path, subfolder) == output_root:
            continue
        gt_path = os.path.join(dataset_path, subfolder, 'gt', 'gt.txt')
        gt = pd.read_csv(gt_path, delimiter=',', header=None,
                         names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'x', 'y', 'conf', 'z'])
        for i in range(1, gt.frame.max() + 1):
            frame_info = gt[gt.frame == i]
            image_path = os.path.join(dataset_path, subfolder, 'img1', f'{i:06}.jpg')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = Image.open(image_path)
            # plt.imshow(image)
            for _, info in frame_info.iterrows():
                bb_top, bb_left, bb_width, bb_height = info.bb_top.astype(int), info.bb_left.astype(
                    int), info.bb_width.astype(int), info.bb_height.astype(int)
                pt1 = (bb_left, bb_top + bb_height)
                pt2 = (bb_left + bb_width, bb_top)
                pt1, pt2 = check_border(pt1, pt2, image)
                # color = (255, 0, 0)
                # overlay = image.copy()
                # cv2.rectangle(overlay, pt1, pt2, color, 3)
                # plt.imshow(overlay)
                # plt.show()
                new_height = (bb_height * (1 + overlap)).astype(int)
                new_width = (bb_width * (1 + overlap)).astype(int)
                new_pt1 = pt1[0] - abs(bb_width - new_width), pt1[1] + abs(bb_height - new_height)
                new_pt2 = pt2[0] + abs(bb_width - new_width), pt2[1] - abs(bb_height - new_height)
                new_pt1, new_pt2 = check_border(new_pt1, new_pt2, image)
                # color = (0, 255, 255)
                # cv2.rectangle(overlay, new_pt1, new_pt2, color, 3)
                # plt.imshow(overlay)
                # plt.show()
                bbox = (new_pt1, new_pt2)
                crop = imcrop(image, bbox)
                if crop.size == 0:
                    continue
                # plt.imshow(crop)
                # plt.show()
                output_path = os.path.join(output_root, subfolder)
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                output_filename = os.path.join(output_path, f'frame_{i:06}' + '_id_{}.jpg'.format(int(info.id)))
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                response = cv2.imwrite(output_filename, crop)
                if response:
                    print('save success for file: {}'.format(output_filename))
                else:
                    print('Save failed for file: {}'.format(output_filename))
            #     print(bbox)
            # print(image_path)

        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='path of dataset to patch', default=os.getcwd())
    parser.add_argument('-o', '--overlap', help='float of overlap for patches', default=0.2)
    parser.add_argument('-op', '--output', help='path tho save output folder',
                        default=None)

    args = parser.parse_args()
    train_path = os.path.join(args.input, 'train')
    run_patcher(train_path, args.overlap, args.output)
