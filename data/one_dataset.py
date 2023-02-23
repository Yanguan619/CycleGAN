from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class OneDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        if type(opt.dataroot) == str:
            self.A_path = [opt.dataroot]
            self.A_img = [Image.open(i).convert('RGB') for i in [opt.dataroot]]
        else:
            self.A_path = [None,]
            self.A_img = [opt.dataroot]
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, idx):
        A_path = self.A_path[idx]
        A_img = self.A_img[idx]
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)
