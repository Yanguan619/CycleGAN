from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """继承Base，补充测试参数
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # 新增参数
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # 重写参数
        parser.set_defaults(model='test')
        parser.set_defaults(no_dropout=True)
        # 自定义参数
        parser.set_defaults(name='monet2photo_pretrained')
        parser.set_defaults(dataroot=r'D:\CycleGAN\datasets\one')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
