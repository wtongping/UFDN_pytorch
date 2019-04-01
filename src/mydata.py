import os
import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image


def LoadDataset(name, root, batch_size, mode, shuffle=True, style=None, attr=None):
    if name == 'face':
        assert style != None
        if mode == 'train':
            return LoadFace(root, style=style, mode='train', batch_size=batch_size,  shuffle=shuffle)
        elif mode =='test':
            return LoadFace(root, style=style, mode='test', batch_size=batch_size,  shuffle=False)


def LoadFace(data_root, batch_size=32, mode='train', style='three_d', shuffle=True):

    transforms_ = [transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    my_dataset = Face(data_root, transforms_=transforms_, mode=mode, style=style)
    return DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)


class Face(Dataset):
    def __init__(self, data_root, transforms_=None, mode='train', style='three_d'):
        self.data_root = data_root
        self.transform = transforms.Compose(transforms_)
        self.image_list = sorted(glob.glob(os.path.join(data_root, '%s/%s' % (mode, style)) + '/*.*'))

    def __getitem__(self, index):
        img = self.transform(Image.open(self.image_list[index % len(self.image_list)]))
        img = img * 2 - 1
        return img

    def __len__(self):
        return len(self.image_list)


