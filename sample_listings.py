from tqdm import tqdm

from bolt.data.utils.zipmap import ZipRead
import bolt.utils.pickle as pickle
import os


class MOTSampleListing(object):
    def __init__(self, *, zip_file, paths, extension='.jpg', subsets=None):
        if subsets is None:
            subsets = []
        self.subsets = subsets
        self.zip_files = zip_file if isinstance(zip_file, list) else [zip_file]
        self.extension = extension
        self.reference_key = next(iter(paths.keys()))
        self.paths = paths
        self.patch_id_map = {}
        self.id_mapping = {}
        self.__init_path_id()

        for key in self.paths.keys():
            if not isinstance(self.paths[key], list):
                self.paths[key] = [self.paths[key]]
            for i in range(len(self.paths[key])):
                if not self.paths[key][i].endswith('/'):
                    self.paths[key][i] += '/'

    def __call__(self, *args, **kwargs):
        file_infos = []

    def __init_path_id(self):
        for zip_file in self.zip_files:
            zm = ZipRead(zip_file)
            if zip_file.endswith('.dszip'):
                metadata = pickle.load(zm.get_file('.metadata/metadata.pickle'))
                file_list = metadata['file_dict'].keys()
            else:
                file_list = zm.list()
            for full_name in tqdm(file_list):
                file_name = os.path.basename(full_name)
                subfolder_name = os.path.basename(os.path.dirname(full_name))
                _id = int(full_name.split('_')[-1].replace(self.extension, ''))
                if subfolder_name in self.subsets:
                    if subfolder_name not in self.patch_id_map:
                        self.patch_id_map[subfolder_name] = {}
                    if subfolder_name not in self.id_mapping:
                        self.id_mapping[subfolder_name] = []
                    self.id_mapping[subfolder_name].append(_id)
                    if _id not in self.patch_id_map[subfolder_name]:
                        self.patch_id_map[subfolder_name][_id] = []
                    self.patch_id_map[subfolder_name][_id].append(full_name)
            for subfolder in self.subsets:
                self.id_mapping[subfolder] = sorted(list(set(self.id_mapping[subfolder])))
        print('Loading datasets complete...')


if __name__ == '__main__':
    data_set_path = r"C:\Users\kap4hi\Downloads\Datasets_zip\2DMOT2015.dszip"
    mot_sapmpler = MOTSampleListing(zip_file=data_set_path,
                                    paths={'path': r'ADL-Rundle-6\frame_000050_id_2.jpg'},
                                    subsets=os.listdir(r'C:\Users\kap4hi\Downloads\2DMOT2015\train'))
