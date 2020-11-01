#!/usr/bin/env python3
"""
This script transforms an image dataset to a dszip file that can be read by the ZipReader. While the script is running
it maintains a temporary file called ``temp_metadata``. If it was interrupted it can continue from where it left off if
the file is found and the target dszip file already exists. This is done by creating a backup pickle for the metadata
(note that this means that this backup is overwritten for each KeyboardInterrupt as well).
 The script can be used with the following command:

``python scripts/create_zip_dataset_by_dir.py <dszip_file_path> <dataset_path> [<dataset_path2> ...]``

Where ``dataset_path`` is the path to the directory containing the files of the dataset.

The following optional parameters can be specified:

    * ``-d <dir_depth>`` Specify the depth of the directory structure of the dataset that will wil transferred to the \
        dszip file. By default the whole path specified in ``dataset_path`` will be kept. \
        Use 0 to add the images directly to the dszip file.
    * ``-u`` Disable the lz4 compression and keep the files in their original format. This option is recommended for \
        datasets in JPG format.
    * ``--jpg_quality`` The quality of the jpg compression. This will only be used if the '-u' flag is also set and \
        the original images were jpgs or videos. Default is 95"
    * ``-c <compression_level>`` Set the compression level for the lz4 algorithm. The Values can be between 0 and 16. \
        Default is 9.
    * ``-v`` Set this if the files are in video format, for example .avi
    * ``-g <config_file_path>`` Pre-process the images with a processing graph. The graph has to be defined in a \
        yml or json config file. A config file that resizes the images would look like this:

        .. code:: yaml

            graph:
                resize:
                    function: bolt.data.processing.image_resize
                    params:
                      output_shape: [1024, 512]
                    inputs: image
                outputs: resize

        If the graph returns None for an Image it is skipped and not added to the dszip file.
    * ``-p <number_of_processes>`` The number of processes that are used to process the images. Default is 1
    * ``-l <path_to_log_file>`` Activate logging to a log file and set the path to the log file
"""
import cv2
import numpy as np
import lz4.frame
from natsort import natsorted
import os
import zipfile
import pickle
import argparse
import sys
from bolt.utils.cfgloader import load_cfg
from bolt.data import Graph
from bolt.data.utils import color_conversion_opencv
from pathlib import Path
from multiprocessing.pool import Pool as Pool
from tqdm import tqdm
from bolt.data import logger
from bolt.utils.log_utils import add_handlers
import tempfile


def get_image_and_metadata(image_path):
    """
    Opens the file at image_path and returns it and it's metadata

    Parameters
    ----------
    image_path: str
        the path to the file

    Return
    ------
    array, dict
        A numpy array containing the data and a dict containing the metadata
    """
    image = cv2.imread(image_path, -1)
    if image is not None:
        image = color_conversion_opencv(image)
        image_metadata = {'shape': image.shape, 'type': image.dtype}
        return image, image_metadata

    else:
        with open(image_path, 'rb') as reader:
            data = reader.read()
            metadata = {}
        return data, metadata


def get_frames_and_metadata(video_path):
    """
    Reads the video at video_path and returns a list containing the frames and a list containing the frame_metadata
    Parameters
    ----------
    video_path: str
        the path to the file

    Return
    ------
    list, list
        a list containing the frames as numpy arrays and a list containing the frame_metadata
    """
    frames = []
    metadata_list = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = color_conversion_opencv(frame)
            frames.append(frame)
            metadata_list.append({'shape': frame.shape, 'type': frame.dtype})
        else:
            break

    cap.release()
    return frames, metadata_list


class ImageVideoZipPreprocessor:
    """
    Functor for create_dszip() for preprocessing images or videos

    Parameters
    ----------
    uncompressed: bool
        whether or not the data should be transformed to np arrays and compressed with lz4
    compression_lvl: int
        the compression level for lz4
    video: bool
        whether the files are in video or image format
    graph_dict: dict, default None
        If this is set all loaded images pre-processed with the given graph
    jpg_quality: int, default 95
        the quality for the jpg compression. Only used if uncompressed is true
    """
    def __init__(self, video, uncompressed, compression_lvl, graph_dict=None, jpg_quality=95):
        self.process_func = self.process_video if video else self.process_image
        self.uncompressed = uncompressed
        self.compression_lvl = compression_lvl
        self.jpg_quality = jpg_quality
        self.graph_dict = graph_dict
        if self.graph_dict is not None:
            self.graph_dict['graph']['inputs'] = ['image']

    def __call__(self, file_path, original_path):
        return self.process_func(file_path, original_path,
            self.uncompressed, self.compression_lvl, self.graph_dict, self.jpg_quality)

    @staticmethod
    def process_image(file_path, original_path, uncompressed, compression_lvl, graph=None, jpg_quality=95):
        """
        Reads the image and the metadata and preprocesses and compresses it if activated

        Parameters
        ----------
        file_path: str
            file path to the image that is used in the dszip file
        original_path: str
            file path to the image on disk
        uncompressed: bool
            whether or not the data should be transformed to np arrays and compressed with lz4
        compression_lvl: int
            the compression level for lz4
        graph: Graph, default None
            processing graph that pre-processes the images
        jpg_quality: int, default 95
            the quality for the jpg compression. Only used if uncompressed is true and the original image was a jpg

        Return
        ------
        dict
            the updated metadata, the image data needed for the add_to_zip function and the file_path
        """
        image, image_metadata = get_image_and_metadata(original_path)

        if graph is not None:
            graph = Graph(graph['graph'], name='Pre-processingGraph')

            if len(image_metadata) > 0:
                image = graph(image=image)
                if image is None:
                    logger.warning("skipping image {} because preprocessing generator returned None".format(original_path))
                    return None, None, None
                image_metadata = {'shape': image.shape, 'type': image.dtype}

        if uncompressed:
            # store Files in their original format in the zip
            if graph is not None and len(image_metadata) > 0:
                # encode changed image data in their original format
                image = color_conversion_opencv(image)
                if Path(original_path).suffix in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                    success, image_data = cv2.imencode(Path(original_path).suffix, image,
                                                    (cv2.IMWRITE_JPEG_QUALITY, jpg_quality))
                else:
                    success, image_data = cv2.imencode(Path(original_path).suffix, image)
                if success:
                    return_data = (file_path, image_data)
                else:
                    logger.error("couldn't encode image {} after pre-processing it".format(original_path))
                    return None, None, None
            else:
                return_data = (file_path, original_path)
            image_metadata['name'] = file_path

        else:
            # change the file format to np and compress it with lz4
            if len(image_metadata) > 0:
                intern_path = file_path.split(Path(file_path).suffix)[0] + '.np.lz4'
                image = image.tobytes()
            else:
                intern_path = file_path + '.lz4'

            compressed_data = lz4.frame.compress(image, compression_lvl)
            return_data = (intern_path, compressed_data)
            image_metadata['name'] = intern_path

        return image_metadata, return_data, file_path

    @staticmethod
    def process_video(file_path, original_path, uncompressed, compression_lvl, graph=None, jpg_quality=95):
        """
        Reads the video and the metadata and preprocesses and compresses it if activated

        Parameters
        ----------
        file_path: str
            file path to the video that is used in the dszip file
        original_path: str
            file path to the video on disk
        uncompressed: bool
            whether or not the data should be transformed to np arrays and compressed with lz4
        compression_lvl: int
            the compression level for lz4
        graph: Graph, default None
            processing graph that pre-processes the videos
        jpg_quality: int, default 95
            the quality for the jpg compression. Only used if uncompressed is true

        Return
        ------
        dict
            the updated metadata
        """
        frame_list, frame_metadata_list = get_frames_and_metadata(original_path)

        if graph is not None:
            graph = Graph(graph['graph'], name='Pre-processingGraph')

            for i in range(len(frame_list)):
                frame_list[i] = graph(image=frame_list[i])
                if frame_list[i] is None:
                    logger.warning("skipping video {} because preprocessing generator returned None".format(original_path))
                    return None, None, None

        return_data = []
        paths = []
        for i in range(len(frame_list)):
            # change the file format to np and compress it with lz4
            frame_file_path = file_path.split('.')[0] + "/{:05d}".format(i)

            if uncompressed:
                frame_list[i] = color_conversion_opencv(frame_list[i])
                frame = frame_list[i].astype(np.uint8)
                frame_metadata_list[i]['type'] = frame.dtype
                success, compressed_data = cv2.imencode('.jpg', frame, (cv2.IMWRITE_JPEG_QUALITY, jpg_quality))
                intern_path = frame_file_path + '.jpg'
            else:
                compressed_data = lz4.frame.compress(frame_list[i].tobytes(), compression_lvl)
                intern_path = frame_file_path + '.np.lz4'
            return_data.append((intern_path, compressed_data))
            frame_metadata_list[i]['name'] = intern_path
            paths.append(frame_file_path)

        return frame_metadata_list, return_data, paths


def get_file_list_by_dir(data_dirs, dir_depth):
    """
    returns a list with the original path to all files in data_dirs and the transformed file_paths with the given depth

    Parameters
    ----------
    data_dirs: list
        list of directories containing the files
    dir_depth: int
        the depth of the directory structure, that should be kept for the paths in the dszip file
    Return
    ------
    list
        a list containing tuples of (original_path, cut_path)
    """
    file_paths = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            logger.error("Failed to create file List. '{}' is not a directory.".format(data_dir))
            raise NotADirectoryError("Failed to create file List. '{}' is not a directory.".format(data_dir))
        for root, dirs, files in os.walk(data_dir):
            split = root.split('/')
            dir_structure = '/'.join(split[-dir_depth:])
            dir_structure = os.path.normpath(dir_structure).replace('../', '')

            for file in files:
                file_path = dir_structure + '/' + file if dir_depth > 0 else file

                file_paths.append((root + '/' + file, file_path))

    return file_paths


def get_file_list_by_file(file_list, dir_depth):
    """
    returns a list with the original path to all files specified in the file_list and a new path that is either
    specified in the file list (separated by ``;``) or by cutting the original path to dir_depth

    Parameters
    ----------
    file_list: str
        path to a file that contains a list of files
    dir_depth: int
        the depth of the directory structure, that should be kept for the paths in the dszip file
    Return
    ------
    list
        a list containing tuples of (original_path, cut_path)
    """
    file_paths = []
    with open(file_list, 'r') as path_reader:
        original_path = path_reader.readline()
        while original_path:
            original_path = original_path.split('\n')[0]

            # check if path for file in dszip is specified
            if ';' in original_path:
                split = original_path.split(';')
                original_path = split[0]
                file_path = split[1]
            else:
                file_path = original_path

            file_path = file_path.split('/')
            file_path = '/'.join(file_path[-dir_depth:])

            file_paths.append((original_path, os.path.normpath(file_path).replace('../', '')))
            original_path = path_reader.readline()

    return file_paths


def add_to_zip(zip_writer, image_data):
    """
    Adds the image data to the zip file

    Parameters
    ----------
    zip_writer: ZipFile
        the zipfile handler
    image_data: tuple
        The first element is the name of the file in the archive.
        The second element is either the path to the original image file or the image data that
        should be added to the zip
    """
    try:
        zip_writer.getinfo(image_data[0])
        logger.info("File already in zip: {}. File is not written again!".format(image_data[0]))
        return False
    except KeyError:
        if isinstance(image_data[1], str):
            zip_writer.write(image_data[1], arcname=image_data[0])
        else:
            zip_writer.writestr(image_data[0], image_data[1])
        logger.info('added {} to {}'.format(image_data[0], zip_writer.filename))
        return True


def create_dszip(data_dirs, dir_depth, dszip_path, file_processor, sort=True, processes=1):
    """
    Creates a dszip file containing a .metadata/metadata.pickle file and the files in data_dirs, either as
    lz4 compressed numpy arrays or in the original format.

    Parameters
    ----------
    data_dirs: list or str
        the paths to the directories that should be added to the dszip file, or the path to a file
        containing a list of files that should be added to the dszip file
    dir_depth: int
        the depth of the directory structure, that should be transferred to the zip file
    dszip_path: str
        the path to the dszip file that should be created
    file_processor: object
        functor for processing file
    sort: bool default: True
        whether the list should be sorted or stay in the order they are given in
    processes: int, default 1
        the number of processes that are used to process the data
    """

    logger.info("Generating file list...")
    if type(data_dirs) is str:
        file_paths = get_file_list_by_file(data_dirs, dir_depth)
    else:
        file_paths = get_file_list_by_dir(data_dirs, dir_depth)
    logger.info("Found {} files for further processing...".format(len(file_paths)))

    if sort:
        # sort by path that will be used in dszip, ignore case
        file_paths = natsorted(file_paths, key=lambda y: y[1].lower())

    backup_pickle_path = dszip_path + ".backup.pickle"

    if os.path.isfile(dszip_path):
        logger.info("Found existing zip file '{}'. New files will be added to existing archive.".format(dszip_path))
        with zipfile.ZipFile(dszip_path, "r") as src_zipfile:
            # Read meta data from original file
            try:
                metadata = pickle.loads(src_zipfile.read(".metadata/metadata.pickle"))
            except KeyError: # Could not find ".metadata/metadata.pickle" in file
                if os.path.isfile(backup_pickle_path):
                    logger.info("No meta information in zip file, reading from '{}' instead."
                                 .format(backup_pickle_path))
                    with open(backup_pickle_path, "rb") as file:
                        metadata = pickle.load(file)
                else:
                    logger.error("ERROR: Found no meta information ('.metadata/metadata.pickle') in file '{}'"
                                  .format(dszip_path))
                    quit()
            # Because we can not delete an element from the original file, we generate a copy of the original file
            # without '.metadata/metadata.pickle
            tmp_file, tmp_dszip_path = tempfile.mkstemp(suffix=os.path.splitext(os.path.basename(dszip_path))[0],
                                                        dir=os.path.dirname(dszip_path))
            os.close(tmp_file)
            with zipfile.ZipFile(tmp_dszip_path, "w") as tgt_zipfile:
                for item in src_zipfile.infolist():
                    if item.filename != ".metadata/metadata.pickle":
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
                logger.info("using {} processes to process the data".format(processes))
                pool_results = []
                for result in tqdm(pool_results):
                    image_metadata, image_data, file_path = result.get()
                    if image_data is None:
                        continue
                    add_img_to_zip_and_metadata(zip_writer, image_data, file_path, metadata, image_metadata)
                pool.close()
                pool.join()
            else:
                logger.info("Starting zipfile generation using 1 process to process the data".format(processes))
                for original_path, file_path in file_paths:
                    image_metadata, image_data, file_path = file_processor(file_path, original_path)

                    add_img_to_zip_and_metadata(zip_writer, image_data, file_path, metadata, image_metadata)
        except (Exception, KeyboardInterrupt, SystemExit) as e:
            # Catch any exception that occurs during writing ...
            logger.error("Script interrupted by following exception: {}".format(e))

            # ... but make sure that meta data is written to be able to resume once problem is fixed
            with open(backup_pickle_path, "wb") as file:
                pickle.dump(metadata, file)
            if processes > 1:
                pool.close()
                pool.join()
            quit()

        pickle_data = pickle.dumps(metadata)
        zip_writer.writestr('.metadata/metadata.pickle', pickle_data)
        logger.info('added .metadata/metadata.pickle to {}'.format(zip_writer.filename))

        if os.path.isfile(backup_pickle_path):
            os.remove(backup_pickle_path)


def add_img_to_zip_and_metadata(zip_writer, image_data, file_path, metadata, image_metadata):
    """
    Writes an image and updates the metadata in the zipmap

    Parameters
    ----------
    zip_writer: ZipFile
        The zipfile
    image_data: ndarray
        The image
    file_path: str
        Path to the zipfile
    metadata: dict
        Metadata dictionary of the zipfile (all images are in there)
    image_metadata: dict
        Metadata of the image
    """
    if isinstance(image_metadata, list):
        for i in range(len(image_metadata)):
            if add_to_zip(zip_writer, image_data[i]):
                metadata['file_list'].append(image_metadata[i])
                metadata['file_dict'][file_path[i]] = image_metadata[i]
    else:
        if add_to_zip(zip_writer, image_data):
            metadata['file_list'].append(image_metadata)
            metadata['file_dict'][file_path] = image_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert a dataset into a dszip file, containing lz4 compressed numpy arrays')
    parser.add_argument('dszip_path', type=str, help="The path to the dszip file that will be created")
    parser.add_argument('dataset_paths',  nargs='+', type=str,
                        help="The paths to the dataset directories that should be added to the dszip file")
    parser.add_argument('-d', '--dir_depth', dest='dir_depth', type=int, default=sys.maxsize,
                        help="The depth of the directory structure to be transferred to the dszip file. "
                             "Choose 0 to discard the directory structure and add the files directly "
                             "to the dszip file. By default the whole directory structure is kept")
    parser.add_argument('-u', '--uncompressed', dest='uncompressed', action='store_true',
                        help="Store the images in the dszip file without compressing them with lz4 and "
                             "keep them in their original data format.")
    parser.add_argument('--jpg_quality', dest='jpg_quality', type=int, default=95,
                        help="The quality of the jpg compression. This will only be used if the '-u' flag is"
                             "also set and the original images were jpgs. Default is 95")
    parser.add_argument('-c', '--compression_lvl', dest='compression_lvl', type=int, default=9,
                        help="The compression level of lz4. Possible Values are between 0 and 16. "
                             "This will only be used if the '-u' flag is not set. Default is 9")
    parser.add_argument('-v', '--video', dest='video', action='store_true',
                        help="Set this if the files are in video format, for example .avi")
    parser.add_argument('-g', '--graph', dest='graph_config', action='store',
                        help="Pre-process the images with a processing graph."
                             "The graph needs to be defined in a config yml or json file")
    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help="The number of processes that are used to process the images. Default is 1")
    parser.add_argument('-l', '--logging', dest='log_path', action='store',
                        help="activate logging to a log file and set the path to the log file")
    args = parser.parse_args()

    if '.dszip' not in args.dszip_path:
        args.dszip_path += '.dszip'

    if args.log_path is not None:
        import logging
        file = logging.FileHandler(filename=args.log_path, mode="w")
        file.setFormatter(logging.Formatter(fmt="{asctime} [{levelname} | {name}] : {message}",
                                            datefmt="%m-%d %H:%M", style="{"))
        file.setLevel(logging.DEBUG)
        add_handlers([file])
        logger = logging.getLogger(__name__)

    graph_dict = None
    if args.graph_config is not None:
        graph_dict = load_cfg(args.graph_config)
        logger.info("Using a graph to pre-process the images")

    file_processor = ImageVideoZipPreprocessor(
        args.video, args.uncompressed, args.compression_lvl, graph_dict, args.jpg_quality)

    create_dszip(args.dataset_paths, args.dir_depth, args.dszip_path, file_processor, processes=args.processes)
