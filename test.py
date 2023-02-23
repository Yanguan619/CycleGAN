import os
import datetime

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
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
    dataset = create_dataset(opt)
    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    # wandb logger
    if opt.use_wandb:
        if not wandb.run:
            wandb_run = wandb.init(
                project=opt.wandb_project_name, name=opt.name, config=opt)
        else:
            wandb.run
        wandb_run._label(repo='CycleGAN')

    # 创建一个html文件：用于对比输入图片与输出文件
    # 1 定义html保存路径
    now_time = datetime.datetime.now().strftime(r'%Y-%m-%d_%H%M%S')
    web_dir = os.path.join(opt.results_dir, opt.name,
                           f'{opt.phase}_{opt.epoch}', now_time)
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = f'{web_dir:s}_iter{opt.load_iter:d}'
    print('creating web directory', web_dir)
    # 2
    webpage = html.HTML(web_dir=web_dir,
                        title=f'Experiment = {opt.name:s}, Phase = {opt.phase:s}, Epoch = {opt.epoch:s}')
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # 模型部分
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # 推理
        visuals = model.get_current_visuals()  # 获取结果图像
        img_path = model.get_image_paths()     # 获取图像路径
        if i % 5 == 0:  # 将图像到一个HTML文件中
            print(f'processing {i:0>4}-th image... {img_path[0]:s}')
        save_images(webpage,
                    visuals,
                    img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize,
                    use_wandb=opt.use_wandb
                    )
    webpage.save()  # 保存HTML文件
