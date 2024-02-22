import argparse
from ast import arg
import logging
from warnings import warn
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from PIL import Image
import random
import os
import colorlog
import numpy as np
import torch

from torchvision import transforms
from torch import Tensor
from lightning.pytorch import LightningModule
from matplotlib import pyplot as plt
import matplotlib

log_colors_config = {
    # 终端输出日志颜色配置
    "DEBUG": "white",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

default_formats = {
    # 日志输出格式
    "log_format": "[%(asctime)s] %(name)s %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s]> %(message)s",
    # 终端输出格式
    "color_format": "%(log_color)s[%(asctime)s] %(name)s %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s]> %(message)s",
}


class Log:
    """
    先创建日志记录器（logging.getLogger），然后再设置日志级别（logger.setLevel），
    接着再创建日志文件，也就是日志保存的地方（logging.FileHandler），然后再设置日志格式（logging.Formatter），
    最后再将日志处理程序记录到记录器（addHandler）
    """

    def __init__(self, log_path=""):
        cur_path = Path.cwd()  # 当前项目路径
        log_path = cur_path / "logs" / log_path  # log_path为存放日志的路径
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)  # 若不存在logs文件夹，则自动创建
        self.__now_time = datetime.now().strftime("%Y-%m-%d")  # 当前日期格式化
        # 收集所有日志信息文件
        self.__all_log_path = log_path.joinpath(self.__now_time + "-all.log")
        # 收集错误日志信息文件
        self.__error_log_path = log_path.joinpath(self.__now_time + "-error.log")
        self.__logger = logging.getLogger()  # 创建日志记录器
        self.__logger.setLevel(logging.DEBUG)  # 设置默认日志记录器记录级别

    @staticmethod
    def __init_logger_handler(log_path):
        """
        创建日志记录器handler，用于收集日志
        :param log_path: 日志文件路径
        :return: 日志记录器
        """
        # 写入文件，如果文件超过1M大小时，切割日志文件，仅保留3个文件
        logger_handler = RotatingFileHandler(
            filename=log_path, maxBytes=1 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        return logger_handler

    @staticmethod
    def __init_console_handle():
        """创建终端日志记录器handler，用于输出到控制台"""
        console_handle = colorlog.StreamHandler()
        return console_handle

    def __set_log_handler(self, logger_handler, level=logging.DEBUG):
        """
        设置handler级别并添加到logger收集器
        :param logger_handler: 日志记录器
        :param level: 日志记录器级别
        """
        logger_handler.setLevel(level=level)
        self.__logger.addHandler(logger_handler)

    def __set_color_handle(self, console_handle):
        """
        设置handler级别并添加到终端logger收集器
        :param console_handle: 终端日志记录器
        :param level: 日志记录器级别
        """
        console_handle.setLevel(logging.DEBUG)
        self.__logger.addHandler(console_handle)

    @staticmethod
    def __set_color_formatter(console_handle, color_config):
        """
        设置输出格式-控制台
        :param console_handle: 终端日志记录器
        :param color_config: 控制台打印颜色配置信息
        :return:
        """
        formatter = colorlog.ColoredFormatter(
            default_formats["color_format"], log_colors=color_config
        )
        console_handle.setFormatter(formatter)

    @staticmethod
    def __set_log_formatter(file_handler):
        """
        设置日志输出格式-日志文件
        :param file_handler: 日志记录器
        """
        formatter = logging.Formatter(
            default_formats["log_format"], datefmt="%a, %d %b %Y %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

    @staticmethod
    def __close_handler(file_handler):
        """
        关闭handler
        :param file_handler: 日志记录器
        """
        file_handler.close()

    def __console(self, level, message):
        """构造日志收集器"""
        all_logger_handler = self.__init_logger_handler(self.__all_log_path)  # 创建日志文件
        error_logger_handler = self.__init_logger_handler(self.__error_log_path)
        console_handle = self.__init_console_handle()

        self.__set_log_formatter(all_logger_handler)  # 设置日志格式
        self.__set_log_formatter(error_logger_handler)
        self.__set_color_formatter(console_handle, log_colors_config)

        self.__set_log_handler(all_logger_handler)  # 设置handler级别并添加到logger收集器
        self.__set_log_handler(error_logger_handler, level=logging.ERROR)
        self.__set_color_handle(console_handle)

        if level == "info":
            self.__logger.info(message)
        elif level == "debug":
            self.__logger.debug(message)
        elif level == "warning":
            self.__logger.warning(message)
        elif level == "error":
            self.__logger.error(message)
        elif level == "critical":
            self.__logger.critical(message)
        self.__logger.removeHandler(all_logger_handler)  # 避免日志输出重复问题
        self.__logger.removeHandler(error_logger_handler)
        self.__logger.removeHandler(console_handle)

        self.__close_handler(all_logger_handler)  # 关闭handler
        self.__close_handler(error_logger_handler)

    def debug(self, message):
        self.__console("debug", message)

    def info(self, message):
        self.__console("info", message)

    def warning(self, message):
        self.__console("warning", message)

    def error(self, message):
        self.__console("error", message)

    def critical(self, message):
        self.__console("critical", message)


def image_path2tensor(path: str):
    pil = Image.open(path).convert("RGB")

    trans = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = trans(pil)
    img = torch.unsqueeze(img, 0)  # 填充一维
    return img


def tensor2pil(x: Tensor):
    image = x.cpu().clone()
    image = image.squeeze(0)  # 压缩一维
    image = transforms.ToPILImage()(image)  # 自动转换为0-255
    return image


# 图像处理
def tensor2np(input_image: Tensor, imtype=np.uint8):
    if len(input_image.size()) == 3:
        input_image = input_image.unsqueeze(0)
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def now_time():
    return datetime.now().strftime(r"%Y-%m-%d-%H%M%S")


def show_img(image_numpy: np.array, aspect_ratio=1.0, save_image=False, save_path=None):
    img_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        img_pil = img_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        img_pil = img_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    if save_image:
        save_dir = Path("logs", "images")
        save_dir.mkdir(parents=True, exist_ok=True)
        img_path = save_dir / str(now_time() + ".jpg")
        img_pil.save(img_path)
        print(f"Image saved in {img_path}")
    else:
        img_pil.show()


def show_imgs(imgs, num=2, titles: list[str] = ["原图", "风格转换后的图片"]):
    matplotlib.rc("font", family="SimHei")
    plt.figure()
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(imgs[i], interpolation="none")
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def show_pic(dataloader, num=4):
    """展示 dataloader 里的 num 张图片"""

    examples = enumerate(dataloader)  # 组合成一个索引序列
    batch_idx, example_data = next(examples)
    plt.figure()
    for i in range(num):
        plt.subplot(2, num // 2, i + 1)
        # plt.tight_layout()
        img = example_data["A"][i]
        print("pic shape:", img.shape)
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img, interpolation="none")
        plt.title(example_data["A_paths"][i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def show_acces(train_loss, train_acc, valid_acc, num_epoch):
    # 对准确率和loss画图显得直观
    from matplotlib import pyplot as plt

    plt.plot(
        1 + np.arange(len(train_loss)),
        train_loss,
        linewidth=1.5,
        linestyle="dashed",
        label="train_loss",
    )
    plt.plot(
        1 + np.arange(len(train_acc)),
        train_acc,
        linewidth=1.5,
        linestyle="dashed",
        label="train_acc",
    )
    plt.plot(
        1 + np.arange(len(valid_acc)),
        valid_acc,
        linewidth=1.5,
        linestyle="dashed",
        label="valid_acc",
    )
    plt.grid()
    plt.xlabel("epoch")
    plt.xticks(range(1, 1 + num_epoch, 1))
    plt.legend()
    plt.show()


def get_config(config) -> dict:
    import yaml

    with open(config, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def get_all_weights(model_name):
    weights_path = Path("weights", model_name)

    if weights_path.exists():
        # 获取文件夹下所有.pth文件（不包括文件夹）
        all_weights = [
            str(f.name)
            for f in Path(weights_path).iterdir()
            if Path(f).is_file() and Path(f).suffix in (".pth", ".pt", ".pkl", ".ckpt")
        ]
        return all_weights
    else:
        warn(f"{weights_path} no find.")
        return None


def get_all_datasets(data_dir="data"):
    datasets_path = Path(data_dir)

    if datasets_path.exists():
        # 获取文件夹下所有.pth文件（不包括文件夹）
        all_datasets = [
            str(f.name) for f in Path(datasets_path).iterdir() if Path(f).is_dir()
        ]
        return all_datasets
    else:
        warn(f"{datasets_path} no find.")
        return None


def find_model_class_using_name(model_name):
    import importlib

    model_filename = f"unit.models.{model_name}"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
            % (model_filename, target_model_name)
        )
        exit(0)
    return model


def model2onnx(
    model: torch.nn.Module,
    input_size=(1, 3, 256, 256),
    onnx_path="model_troch_export.onnx",
):
    x = torch.randn(*input_size)

    # a necessary fix, applicable only for Efficientnet
    # model.model.set_swish(memory_efficient=False)
    # model_quantization = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.Conv2d, torch.nn.ReLU}, dtype=torch.quint8
    # )# 只是量化模型的权重参数，而不是整个模型
    torch.onnx.export(
        model,  # model being run
        ##since model is in the cuda mode, input also need to be
        x,  # model input (or a tuple for multiple inputs)
        onnx_path,  # where to save the model (can be a file or file-like object)
        # export_params=True,  # store the trained parameter weights inside the model file
        # opset_version=10,  # the ONNX version to export the model to
        # do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["image"],  # the model's input names
        # output_names=["output"],  # the model's output names
        # dynamic_axes={
        #     "input": {0: "batch_size"},  # variable lenght axes
        #     "output": {0: "batch_size"},
        # },
    )
    # import onnx
    # import onnx.numpy_helper
    # # Step 2: Compute output ranges
    # onnx_model = onnx.load(onnx_path)
    # model_outputs = [output.name for output in onnx_model.graph.output]
    # output_ranges = {}
    # for output in model_outputs:
    #     tensor = onnx.numpy_helper.to_array(onnx_model.graph.get_tensor(output))
    #     output_ranges[output] = (float(tensor.min()), float(tensor.max()))

    # Step 3: Quantize the model parameters
    # from onnxruntime.quantization import quantize, QuantizationMode
    # quantized_model = quantize(
    #     onnx_path,
    #     'quantized_model.onnx',
    #     quant_config=
    # )


def model2onnx_lightning(
    model: LightningModule, onnx_path="model_lightnining_export.onnx"
):
    ## a necessary fix, applicable only for Efficientnet
    # model.set_swish(memory_efficient=False)
    model.to_onnx(
        file_path=onnx_path,
        input_sample=model.example_input_array.to("cuda"),
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def pytorch_out(model: torch.nn.Module, input) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(input)
    return output


def test_onnx(model, onnx_path, input_size: tuple):
    try:
        import onnxruntime
        from onnxruntime.datasets import get_example
    except ImportError or ImportWarning:
        return
    torch.manual_seed(66)
    # example_input = torch.randn(*input_size, device="cpu")
    example_input = image_path2tensor(r"D:\projects\CycleGAN-1.0\unit\inputs\horse.jpg")

    def to_np(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnx_session = onnxruntime.InferenceSession(onnx_path)
    input_name = onnx_session.get_inputs()[0].name

    # np.random.randn(*input_size).astype(np.float32)
    onnx_input = {input_name: to_np(example_input)}
    onnx_out = torch.from_numpy(np.array(onnx_session.run(None, onnx_input)[0]))
    print("==============>")
    print(onnx_out)
    print(onnx_out.shape)
    print("==============>")
    torch_out = pytorch_out(model, example_input).detach()  # fc输出是二维 列表
    print(torch_out)
    print(torch_out.shape)

    print("===================================>")
    print("输出结果验证小数点后五位是否正确,都变成一维np")
    torch_out_res = torch_out.numpy().flatten()
    onnx_out_flatten = np.array(onnx_out).flatten()
    pytor = np.array(torch_out_res, dtype="float32")  # need to float32
    onn = np.array(onnx_out_flatten, dtype="float32")  ##need to float32
    np.testing.assert_almost_equal(pytor, onn, decimal=4)  # 精确到小数点后5位，验证是否正确，不正确会自动打印信息
    print("🎉 onnx 和 pytorch 结果一致.")
    print("Exported model has been executed decimal=5 and the result looks good!")
    show_img(tensor2np(torch_out))
    show_img(tensor2np(onnx_out))


def test():
    from unit.models.cycle_gan import CycleGANModel

    model = CycleGANModel()
    print(model.net_G_A)


def test2(onnx_path="model_troch_export.onnx", input_size=(1, 3, 256, 256)):
    from unit.models.cycle_gan import CycleGANModel
    from unit.trainer.cycle_gan import load

    model = CycleGANModel()
    load(model.net_G_A, r"D:\projects\CycleGAN-1.0\weights\cycle_gan\horse2zebra.pth")
    model2onnx(model=model.net_G_A, input_size=input_size, onnx_path=onnx_path)
    test_onnx(model, onnx_path, input_size)


def fix_seed(seed: int = 77) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) # 检测是否使用了随机算法，使用了会报错
    print(f"Random seed set as {seed}")


def print_args(args: argparse.ArgumentParser):
    print("Options:")
    for k, v in vars(args.parse_args()).items():
        default = args.get_default(k)
        comment = ""
        if v != default:
            comment = "\t[default: %s]" % str(default)
        print("\t{:<15}: {:<20}{}".format(str(k), str(v), comment))
