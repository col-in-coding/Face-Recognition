import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CASIAWebFace(Dataset):
    def __init__(self, root, file_list, transform=None):
        self.root = root
        self.transform = transform
        self.loader = cv2.imread

        image_list = []
        label_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))

        # random flip with ratio 0.5
        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            img = cv2.flip(img, 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return (img, label)

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # validation dataset
    root = "/home/coin/Documents/face_recognition/chapter3/data/unlabelled"
    file_list = "/home/coin/Documents/face_recognition/hw5/data_list.txt"
    dataset = CASIAWebFace(root, file_list, transform=transform)
    trainloader = DataLoader(dataset, batch_size=2, shuffle=True,
                             num_workers=1, drop_last=False)

    # for data in trainloader:
    #     print(data[0].shape)
