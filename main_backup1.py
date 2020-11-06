import argparse
import os
import pickle
import sys
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path
from time import time
import lz4.frame
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted
from tqdm import tqdm


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
    overlap = float(overlap)
    if output_root is None:
        output_root = os.path.join(dataset_path, 'output')
    if not os.path.isdir(output_root):
        os.makedirs(output_root)
    for subfolder in os.listdir(dataset_path):
        if subfolder == 'output':
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


def get_image_and_metadata(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_metadata = {'shape': image.shape, 'type': image.dtype}
        return image, image_metadata
    else:
        with open(image_path, 'rb') as reader:
            data = reader.data()
            metadata = {}
            return data, metadata


def process_image(file_path, original_path, uncompressed, compression_lvl, jpeg_quality):
    image, image_metadata = get_image_and_metadata(original_path)
    return_data = None
    if uncompressed:
        if len(image_metadata) > 0:
            if Path(original_path).suffix in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                success, image_data = cv2.imencode(Path(original_path).suffix, image,
                                                   (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality))
            else:
                success, image_data = cv2.imencode(Path(original_path).suffix, image)
            if success:
                return_data = (file_path, image_data)
            else:
                print('cannot encode image {} after pre-processing it'.format(original_path))
        else:
            return_data = (file_path, original_path)
        image_metadata['name'] = file_path
    else:
        if len(image_metadata) > 0:
            intern_path = file_path.split(Path(file_path).suffix)[0] + '.np.lz4'
            image = image.tobytes()
        else:
            intern_path = file_path + '.lz4'

        compressed_data = lz4.frame.compress(image, compression_lvl)
        return_data = (intern_path, compressed_data)
        image_metadata['name'] = intern_path

    return image_metadata, return_data, file_path


def get_file_list(dataset_path, dir_depth):
    file_paths = []
    for data_dir in dataset_path:
        if not os.path.isdir(data_dir):
            raise NotADirectoryError('Failed to create file list {} is not a directory'.format(data_dir))
        for root, dirs, files in os.walk(data_dir):
            split = root.split('/')
            dir_structure = '/'.join(split[-dir_depth:])
            dir_structure = os.path.normpath(dir_structure).replace('../', '')

            for file in files:
                file_path = dir_structure + '/' + file if dir_depth > 0 else file
                file_paths.append((root + '/' + file, file_path))

    return file_paths


def add_to_zip(zip_writer, image_data):
    try:
        zip_writer.getinfo(image_data[0])
        print('file already exist in zip {}'.format(image_data[0]))
        return False
    except KeyError:
        if isinstance(image_data[1], str):
            arc_name = os.path.join(os.path.basename(os.path.dirname(image_data[0])), os.path.basename(image_data[0]))
            zip_writer.write(image_data[1], arcname=arc_name)
        else:
            arc_name = os.path.join(os.path.basename(os.path.dirname(image_data[0])), os.path.basename(image_data[0]))
            zip_writer.writestr(arc_name, image_data[1])
        print('added {} to {} '.format(image_data[0], zip_writer.filename))
        return True


def add_image_to_zip_and_metadata(zip_writer, image_data, file_path, metadata, image_metadata):
    if isinstance(image_metadata, list):
        for i in range(len(image_metadata)):
            if add_to_zip(zip_writer, image_metadata[i]):
                file_name = os.path.join(os.path.basename(os.path.dirname(image_metadata[i]['name'])),
                                         os.path.basename(image_metadata[i]['name']))
                metadata['file_list'].append(file_name)
                metadata['file_dict'][file_path[i]] = file_name
    else:
        if add_to_zip(zip_writer, image_data):
            file_name = os.path.join(os.path.basename(os.path.dirname(image_metadata['name'])),
                                     os.path.basename(image_metadata['name']))
            metadata['file_list'].append(file_name)
            metadata['file_dict'][file_path] = file_name


def create_dszip(data_dirs, dszip_path, dir_depth, uncompressed=True, compression_lvl=9, jpeg_quality=95, processes=1):
    file_paths = get_file_list(data_dirs, dir_depth)
    print('Found {} files for further processing...'.format(len(file_paths)))
    file_paths = natsorted(file_paths, key=lambda y: y[1].lower())
    backup_pickle_path = dszip_path + '.backup.pickle'
    if os.path.isfile(dszip_path):
        print('Found existing zip file {}. New files will be added to existing archive. '.format(dszip_path))
        with zipfile.ZipFile(dszip_path, 'r') as src_zipfile:
            try:
                metadata = pickle.loads(src_zipfile.read('.metadata/metadata.pickle'))
            except KeyError:
                if os.path.isfile(backup_pickle_path):
                    print('No metadata information found reading form {} instead'.format(backup_pickle_path))
                    with open(backup_pickle_path, 'rb') as file:
                        metadata = pickle.load(file)
                else:
                    print('Error: Found no metadata information')
                    quit()
            # print(os.path.dirname(os.path.dirname(dszip_path)))
            # tmp_path = os.path.join(data_dirs[0])
            # print(os.path.splitext(os.path.basename(dszip_path))[0])
            tmp_file, tmp_dszip_path = tempfile.mkstemp(suffix=os.path.splitext(os.path.basename(dszip_path))[0],
                                                        dir=os.path.dirname(dszip_path))
            os.close(tmp_file)
            with zipfile.ZipFile(tmp_dszip_path, 'w') as tgt_zipfile:
                for item in src_zipfile.infolist():
                    if item.filename != '.metadata/metadata.pickle':
                        buffer = src_zipfile.read(item.filename)
                        tgt_zipfile.writestr(item, buffer)
        os.remove(dszip_path)
        os.rename(tmp_dszip_path, dszip_path)

    else:
        metadata = {'file_list': [], 'file_dict': {}}

    with zipfile.ZipFile(dszip_path, 'a', compression=zipfile.ZIP_STORED) as zip_writer:
        try:
            if processes > 1:
                pool = Pool(processes)
                pool_results = []
                for result in tqdm(pool_results):
                    image_metadata, image_data, file_path = result.get()
                    if image_data is None:
                        continue
                    add_image_to_zip_and_metadata(zip_writer, image_data, file_path, metadata, image_metadata)
                pool.close()
                pool.join()
            else:
                print('starting zipfile generation using 1 process to process the data')
                for original_path, file_path in file_paths:
                    image_metadata, image_data, file_path = process_image(file_path, original_path, uncompressed,
                                                                          compression_lvl, jpeg_quality)
                    add_image_to_zip_and_metadata(zip_writer, image_data, file_path, metadata, image_metadata)
        except (Exception, KeyboardInterrupt, SystemExit) as e:
            print('Script interrupted by following exception: {}'.format(e))
            with open(backup_pickle_path, 'wb') as file:
                pickle.dump(metadata, file)
            if processes > 1:
                pool.close()
                pool.join()
            quit()

        pickle_data = pickle.dumps(metadata)
        zip_writer.writestr('.metadata/metadata.pickle', pickle_data)
        print('added .metadata/metadata.pickle to {}'.format(zip_writer.filename))
        if os.path.isfile(backup_pickle_path):
            os.remove(backup_pickle_path)


if __name__ == '__main__':
    start_time = time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='path of dataset to patch', default=os.getcwd())
    parser.add_argument('-o', '--overlap', help='float of overlap for patches', default=0.2)
    parser.add_argument('-op', '--output', help='path tho save output folder',
                        default=None)
    parser.add_argument('-ds', '--dszip_path', help='Path of storing dszip file', default=None)
    parser.add_argument('-u', '--uncompressed', dest='uncompressed', action='store_true',
                        help="Store the images in the dszip file without compressing them with lz4 and "
                             "keep them in their original data format.", default=True)
    parser.add_argument('--jpg_quality', dest='jpg_quality', type=int, default=95,
                        help="The quality of the jpg compression. This will only be used if the '-u' flag is"
                             "also set and the original images were jpgs. Default is 95")
    parser.add_argument('-c', '--compression_lvl', dest='compression_lvl', type=int, default=9,
                        help="The compression level of lz4. Possible Values are between 0 and 16. "
                             "This will only be used if the '-u' flag is not set. Default is 9")
    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help="The number of processes that are used to process the images. Default is 1")
    parser.add_argument('-d', '--dir_depth', dest='dir_depth', type=int, default=sys.maxsize,
                        help="The depth of the directory structure to be transferred to the dszip file. "
                             "Choose 0 to discard the directory structure and add the files directly "
                             "to the dszip file. By default the whole directory structure is kept")

    args = parser.parse_args()
    dataset_name = os.path.basename(args.input)
    train_path = os.path.join(args.input, 'train')
    run_patcher(train_path, args.overlap, args.output)
    data_dirs = [os.path.join(train_path, 'output')]
    dszip_path = args.dszip_path
    if dszip_path is None:
        dszip_path = os.path.join(args.input, dataset_name + '.dszip')
    if '.dszip' not in dszip_path:
        dszip_path += '.dszip'

    create_dszip(data_dirs, dszip_path, dir_depth=args.dir_depth, compression_lvl=args.compression_lvl,
                 uncompressed=args.uncompressed, processes=args.processes, jpeg_quality=args.jpg_quality)
    print('processing completed total time taken: {}s'.format(time() - start_time))
