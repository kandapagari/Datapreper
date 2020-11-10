import random

from tqdm import tqdm

from bolt.data.utils.zipmap import ZipRead
import bolt.utils.pickle as pickle
import os


class MOTSampleListing:
    def __init__(self, *, zip_file, extension='.jpg', subsets=None):
        if subsets is None:
            subsets = []
        self.subsets = subsets
        self.zip_files = zip_file if isinstance(zip_file, list) else [zip_file]
        self.extension = extension
        self.patch_id_map = {}
        self.id_mapping = {}
        self.__init_path_id()

    def __call__(self, *args, **kwargs):
        file_infos = []
        for zip_file in self.zip_files:
            for subfolder in self.patch_id_map.keys():
                for _id in self.id_mapping[subfolder]:
                    for i, file in enumerate(self.patch_id_map[subfolder][_id]):
                        if i + 1 == len(self.patch_id_map[subfolder][_id]): break
                        image1 = file
                        image2 = self.patch_id_map[subfolder][_id][i + 1]
                        random_id = random.choice([a for a in self.id_mapping[subfolder] if a is not _id])
                        image3 = random.choice(self.patch_id_map[subfolder][random_id])
                        file_infos.append(
                            {'zip': zip_file, 'image1': image1, 'image2': image2, 'image3': image3, 'ids': [1, 1, 0]})
        return file_infos

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
                    self.patch_id_map[subfolder_name][_id].append(os.path.join(subfolder_name, file_name))
            for subfolder in self.subsets:
                self.id_mapping[subfolder] = sorted(list(set(self.id_mapping[subfolder])))
                for _id in self.patch_id_map[subfolder]:
                    self.patch_id_map[subfolder][_id] = sorted(self.patch_id_map[subfolder][_id])
        print('Loading datasets complete...')


if __name__ == '__main__':
    data_set_path = r"C:\Users\kap4hi\Downloads\Datasets_zip\2DMOT2015.dszip"
    mot_sampler = MOTSampleListing(zip_file=data_set_path,
                                   subsets=os.listdir(r'C:\Users\kap4hi\Downloads\2DMOT2015\train'))
    sampling_list = mot_sampler()
