import os
import torch
from torch.utils.data import Dataset, DataLoader
# import glob
from torchvision import transforms

from PIL import Image

from common import representation
from common import event


# import random


class EventVoxel(Dataset):
    def __init__(self, data_root, is_hsergb, is_training, is_validation, number_of_time_bins, number_of_skips=1):
        """
            Aims to create an EventVoxel object.
            Inputs,
                data_root: Root path for the dataset
                is_hsergb: Indicator for the event-frame dataset False if bs-ergb dataset intedted to be used.
                is_training: Indicates whether the dataset is for training.
                is_validation: Indicates whether the dataset is for validation. ( works if is_training set to false)
                number_of_time_bins: Indicates TWO of the event files' temporal seperation in voxel 
                number_of_skips: No functionality for this version of the class.
            Outputs,
                images = 4 image with 2 leftmost 2 rightmost frame neighbors of the ground truth ( in 4x3x256x256 )
                voxel grid = 12 channel voxel grid representation of 2 leftmost 2 rightmost event neighbors of the ground truth ( in 4x3x256x256 )        
                ground truth = ground truth image ( in 3x256x256 )
        """
        self.data_root = data_root
        self.is_training = is_training  # if True it is training set
        self.is_validation = is_validation  # if True it is validation set
        self.is_hsergb = is_hsergb
        self.number_of_time_bins = number_of_time_bins
        self.number_of_skips = number_of_skips

        if is_hsergb:
            train_fn = os.path.join(self.data_root, 'data_set_hsergb.txt')
        else:
            train_fn = os.path.join(self.data_root, 'set_data_bsergb_training.txt')
            test_fn = os.path.join(self.data_root, 'set_data_bsergb_test.txt')
            valid_fn = os.path.join(self.data_root, 'set_data_bsergb_validation.txt')

        if is_hsergb:
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()  # TODO split to a training set
        else:
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()
            with open(test_fn, 'r') as f:
                self.testlist = f.read().splitlines()
            with open(valid_fn, 'r') as f:
                self.validationlist = f.read().splitlines()

        self.transforms = transforms.Compose([
           transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):

        """
        Every pass returns an image of ground truth , 4 events (2 normal 2 reversed in a voxel of 12 bins) and right,left images
        Total 3 images with 9 channel 1 voxel of 12 channel is obtained.
        """

        if self.is_hsergb:
            raw_data = self.trainlist[index].split(" ")
        else:
            if self.is_training:
                raw_data = self.trainlist[index].split(" ")
            elif self.is_validation:
                raw_data = self.validationlist[index].split(" ")
            else:
                raw_data = self.testlist[index].split(" ")
        raw_data.pop()  # this is for the blank at the end of a line
        raw_data = [os.path.join(self.data_root, raw_) for raw_ in raw_data]
        list_of_images = raw_data[:4]
        gt_path = raw_data[4]
        list_of_events = raw_data[5:]

        # the_pair = random.randint(2, len(list_of_events)-3) #Since the 5 image 4 files of events are selected last 2 and first 2 can not be the ground truth image ( center image to be predicted )

        gt = Image.open(gt_path)  # since events are sparse an image is fetched in order to have HxW information.

        W, H = gt.size

        events_left = event.EventSequence.from_npz_files(
            list_of_filenames=list_of_events[:3],
            image_height=H,
            image_width=W,
            hsergb=self.is_hsergb,
        )

        voxel_left = representation.to_voxel_grid(events_left, nb_of_time_bins=self.number_of_time_bins)

        events_right = event.EventSequence.from_npz_files(
            list_of_filenames=list_of_events[3:],
            image_height=H,
            image_width=W,
            hsergb=self.is_hsergb,
        )
        events_right.reverse()

        voxel_right = representation.to_voxel_grid(events_right, nb_of_time_bins=self.number_of_time_bins)

        voxel = torch.cat((voxel_left, voxel_right))

        images = [Image.open(pth) for pth in list_of_images]

        T = self.transforms

        images = [T(img_) for img_ in images]
        gt = T(gt)

        voxel = torch.reshape(voxel, (4, self.number_of_time_bins // 2, H, W))  # reshape to have a 4x3xHxW tensor
        #V = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)),
        # transforms.ToTensor()])  # Voxel grid transforms for 256 #TODO random crop size is subject to change.

        voxel = torch.nn.functional.interpolate(voxel, size=(256, 256))
        # V = transforms.Compose([transforms.ToTensor()])  # Voxel grid transforms for 256 #TODO random crop size is subject to change.

        #voxel = [V(vox_) for vox_ in voxel[:]]
        voxel = list(voxel)
        return images, voxel, gt ,gt_path # note that all three are torch tensor

    def __len__(self):
        if self.is_training:
            return len(self.trainlist)
        elif self.is_validation:
            return len(self.validationlist)
        else:
            return len(self.testlist)


# def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
#     if mode == 'train':
#         is_training = True
#     else:
#         is_training = False
#     dataset = EventVoxel(data_root, is_training=is_training)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    dataset = EventVoxel("../BS-ERGB", is_hsergb=False, is_training=True, is_validation=False, number_of_time_bins=6)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
