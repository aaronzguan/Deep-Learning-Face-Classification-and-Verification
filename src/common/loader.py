import torch
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from glob import glob
from .util import face_ToTensor


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_aug=False):
        super(TrainingDataset, self).__init__()
        self.args = args
        self.is_aug = is_aug
        # self.faces_path = glob(self.args._data_root + 'train_data/medium/*/*')
        self.faces_path = glob(self.args._data_root + 'train_data/*/*/*')
        self.targets, self.num_class = self.get_targets()

    def __getitem__(self, index):
        face = cv2.imread(self.faces_path[index])
        face = cv2.resize(face, (self.args.image_size, self.args.image_size), cv2.INTER_CUBIC)

        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)

        return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return len(self.faces_path)

    def get_targets(self):
        # class_id = [path.split(self.args._data_root + 'train_data/medium/')[1].split('/')[0] for path in self.faces_path]
        class_id = [path.split(self.args._data_root + 'train_data/')[1].split('/')[1] for path in self.faces_path]
        targets = np.asarray(np.int32(class_id)).reshape(-1, 1)
        unique_class_id = np.unique(targets)
        num_class = unique_class_id.shape[0]
        return targets, num_class


class IdentificationDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_testing=False):
        super(IdentificationDataset, self).__init__()
        self.args = args
        self.is_testing = is_testing
        if self.is_testing:
            with open(os.path.join(self.args._data_root, self.args._test_identification_order)) as f:
                self.order_lines = f.readlines()
            self.num_faces = len(self.order_lines)
        else:
            # self.faces_path = glob(self.args._data_root + self.args._val_identification_set +'/medium/*/*')
            self.faces_path = glob(self.args._data_root + self.args._val_identification_set + '/*/*/*')
            self.targets, self.num_class = self.get_targets()
            self.num_faces = len(self.faces_path)

    def __getitem__(self, index):
        if self.is_testing:
            img_name = self.order_lines[index].replace('\n', '')
            face = cv2.imread(os.path.join(self.args._data_root, self.args._test_identification_set, img_name))
            label = np.int32(0).reshape(1)  # Dummy label
        else:
            face = cv2.imread(self.faces_path[index])
            label = self.targets[index]

        face = cv2.resize(face, (self.args.image_size, self.args.image_size), cv2.INTER_CUBIC)

        return face_ToTensor(face), torch.LongTensor(label)

    def __len__(self):
        return self.num_faces

    def get_targets(self):
        class_id = [path.split(self.args._data_root + self.args._val_identification_set + '/')[1].split('/')[1] for path in self.faces_path]
        # class_id = [path.split(self.args._data_root + self.args._val_identification_set + '/medium/')[1].split('/')[0] for path in self.faces_path]
        targets = np.asarray(np.int32(class_id)).reshape(-1, 1)
        unique_class_id = np.unique(targets)
        num_class = unique_class_id.shape[0]
        return targets, num_class


class VerificationDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_testing=False):
        super(VerificationDataset, self).__init__()
        self.args = args
        self.is_testing = is_testing
        if self.is_testing:
            with open(os.path.join(self.args._data_root, self.args._test_verification_pairs)) as f:
                self.pair_lines = f.readlines()
        else:
            with open(os.path.join(self.args._data_root, self.args._val_verification_pairs)) as f:
                self.pair_lines = f.readlines()
        self.num_faces = len(self.pair_lines)

    def __getitem__(self, index):
        p = self.pair_lines[index].replace('\n', '').split(' ')

        if self.is_testing:
            face1 = cv2.imread(self.args._data_root + os.path.join(self.args._test_verification_set, p[0]))
            face2 = cv2.imread(self.args._data_root + os.path.join(self.args._test_verification_set, p[1]))
            sameflag = np.int32(-1).reshape(1)  # Dummy label
        else:
            face1 = cv2.imread(self.args._data_root + os.path.join(self.args._val_verification_set, p[0]))
            face2 = cv2.imread(self.args._data_root + os.path.join(self.args._val_verification_set, p[1]))
            sameflag = np.int32(p[2]).reshape(1)

        face1 = cv2.resize(face1, (self.args.image_size, self.args.image_size), cv2.INTER_CUBIC)
        face2 = cv2.resize(face2, (self.args.image_size, self.args.image_size), cv2.INTER_CUBIC)
        face1_flip = cv2.flip(face1, 1)
        face2_flip = cv2.flip(face2, 1)

        return face_ToTensor(face1), face_ToTensor(face2), \
               face_ToTensor(face1_flip), face_ToTensor(face2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pair_lines)


class get_loader():
    def __init__(self, args, name, batch_size, is_aug=True, shuffle=True, drop_last=False):
        if name == 'training':
            dataset = TrainingDataset(args, is_aug=is_aug)
            num_class = dataset.num_class
            num_faces = None
        elif name == 'val_identification':
            dataset = IdentificationDataset(args, is_testing=False)
            num_class = dataset.num_class
            num_faces = None
        elif name == 'test_identification':
            dataset = IdentificationDataset(args, is_testing=True)
            num_class = None
            num_faces = None
            shuffle = False
        elif name == 'val_verification':
            dataset = VerificationDataset(args, is_testing=False)
            num_class = None
            num_faces = dataset.num_faces
        elif name == 'test_verification':
            dataset = VerificationDataset(args, is_testing=True)
            num_class = None
            num_faces = None
            shuffle = False

        self.dataloader = DataLoader(dataset=dataset, num_workers=4, batch_size=batch_size,
                                     pin_memory=False, shuffle=shuffle, drop_last=drop_last)
        self.num_class = num_class
        self.num_faces = num_faces
        self.train_iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.train_iter)
        except:
            self.train_iter = iter(self.dataloader)
            data = next(self.train_iter)
        return data