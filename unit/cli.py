"""
## 启动环境
`conda activate yanguan`

## 开始
- 启动web `python -m unit.cli web`
- 启动web `python -m unit.cli web --share`
- 推理图片并展示 `python -m unit.cli detect --show_image`
- 推理图片并保存 `python -m unit.cli detect --save_image`
- 训练CycleGAN模型 `python -m unit.cli train`

## More examples
- `python -m unit.cli detect --source unit\inputs\monet.jpg --style monet2photo.pth --show_image`
- `python -m unit.cli train --data_name horse2waterink --tensorboard`
"""

import argparse
from unit.utils import print_args


def detect_(args):
    from unit.detect import detect

    detect(args.image, args.styles, args.show_image, args.save_image)


def train(args):
    from unit.trainer import cycle_gan

    cycle_gan.main(args.data_name, args.tensorboard)


def web(args):
    from unit import app

    app.start(args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(help="commands")
    # based on app.py
    web_parser = subparsers.add_parser(name="web", help="web server")
    web_parser.add_argument("--share", action="store_true", help="gradio arg")
    web_parser.set_defaults(func=web)
    # based on detect.py
    detect_parser = subparsers.add_parser(name="detect", help="detect")
    detect_parser.add_argument("--image", type=str, help="input image")
    detect_parser.add_argument("--styles", type=str, help="transfer style")
    detect_parser.add_argument("--show_image", action="store_true", help="detect arg")
    detect_parser.add_argument("--save_image", action="store_true", help="detect arg")
    detect_parser.set_defaults(func=detect_)
    # based on trainer package
    # TODO 为了使用LightningCli
    train_parser = subparsers.add_parser(name="train", help="train model")
    train_parser.add_argument("--data_name", required=True, help="file path")
    train_parser.add_argument("--tensorboard", action="store_true")
    train_parser.set_defaults(func=train)

    print_args(parser)
    args = parser.parse_args()
    args.func(args)
