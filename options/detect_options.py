from .test_options import TestOptions


class DetectOptions(TestOptions):
    """继承Test
    填写推理图片路径、推理风格
    """

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        parser.add_argument('--save_fake', type=bool,
                            default=False, help='保存fake图片')
        # 重写参数
        parser.set_defaults(name='horse2zebra_pretrained')
        parser.set_defaults(
            dataroot=r'D:\CycleGAN\datasets\one\n02381460_11.jpg')
        return parser
