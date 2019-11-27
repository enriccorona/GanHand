import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, mode):
        if dataset_name == 'obman':
            from data.dataset_obman import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'contactdb':
            from data.dataset_contactdb import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'double_contactdb':
            from data.double_dataset_contactdb import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'rgb_contactdb':
            from data.rgb_contactdb import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'test_poseestimation_voxels':
            from data.dataset_test_poseestimation_voxels import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'test_poseestimation':
            from data.dataset_test_poseestimation import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_voxels':
            from data.dataset_obman_poseestimation_voxels import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_efficientvoxels':
            from data.dataset_obman_poseestimation_efficientvoxels import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation':
            from data.dataset_obman_poseestimation import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_doublefaces':
            from data.dataset_obman_poseestimation_doublefaces import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_onlyhand':
            from data.dataset_obman_poseestimation_onlyhand import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_onlyhand_doublefaces':
            from data.dataset_obman_poseestimation_onlyhand_doublefaces import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_onlyhand_doublefaces_wnoise':
            from data.dataset_obman_poseestimation_onlyhand_doublefaces_wnoise import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_onlyhand_doublefaces_rescaled':
            from data.dataset_obman_poseestimation_onlyhand_doublefaces_rescaled import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_poseestimation_onlyhand_doublefaces_rescaled_wnoise':
            from data.dataset_obman_poseestimation_onlyhand_doublefaces_rescaled_wnoise import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'dataset_real_imgs':
            from data.dataset_real_imgs import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_depth':
            from data.dataset_obman_depth import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_classification':
            from data.dataset_obman_classification import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_classification_wrot':
            from data.dataset_obman_classification_wrot import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'obman_classification_norot':
            from data.dataset_obman_classification_norot import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'double_obman_classification_norot':
            from data.double_dataset_obman_classification_norot import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'double_obman_classification_norot_normingfurther':
            from data.double_dataset_obman_classification_norot_normingfurther import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'ycb_synth':
            from data.ycb_synth import Dataset
            dataset = Dataset(opt, mode)

## YCB DATASET:
        elif dataset_name == 'ycb_real':
            from data.ycb_real import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'ycb_real_2':
            from data.ycb_real_2 import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'ycb_real_complete_scene':
            from data.ycb_real_complete_scene import Dataset
            dataset = Dataset(opt, mode)

## OTHERS:
        elif dataset_name == 'rgb_obman':
            from data.rgb_obman import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'rgb_obman_mask':
            from data.rgb_obman_mask import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'rgb_obman_test':
            from data.rgb_obman_test import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'rgb_obman_classification_norot_resamplevertices':
            from data.rgb_obman_classification_norot_resamplevertices import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'rgb_obman_classification_norot_resamplevertices_complete':
            from data.rgb_obman_classification_norot_resamplevertices_complete import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'double_obman_classification_norot_resamplevertices':
            from data.double_dataset_obman_classification_norot_resamplevertices import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'double_obman_classification_norot_resamplevertices_complete':
            from data.double_dataset_obman_classification_norot_resamplevertices_complete import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'double_obman_classification_norot_resamplevertices_complete_fullsizeimgs':
            from data.double_dataset_obman_classification_norot_resamplevertices_complete_fullsizeimgs import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'debug':
            from data.dataset_obman_debug import Dataset
            dataset = Dataset(opt, mode)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, mode):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._mode = mode
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images
