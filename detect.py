from options.detect_options import DetectOptions
from data.one_dataset import OneDataset
from models import create_model
from util import util, html
import os
from pathlib import Path
# 定义参数
opt = DetectOptions().parse()  # get test options

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
# disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.serial_batches = True
# no flip; comment this line if results on flipped images are needed.
opt.no_flip = True
# no visdom display; the test code saves the results to a HTML file.
opt.display_id = -1
# create a dataset given opt.dataset_mode and other options


#  加载模型
def load_model():
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model


#  加载数据
def load_data():
    dataset = OneDataset(opt)
    return dataset


#  推理
def detect():
    model = load_model()
    dataset = load_data()
    for _, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # 推理
        visuals = model.get_current_visuals()['fake']  # 获取结果图像
        img = util.tensor2im(visuals)

        return img


img = detect()

if __name__ == '__main__':
    if opt.save_fake:
        # fake 保存路径
        fake_path = Path(opt.dataroot)
        save_path = fake_path.joinpath(
            fake_path.parent, 'fake_' + fake_path.name)
        util.save_image(img, save_path)
        print('results_path: ', save_path)
    else:
        util.show_image(img)
