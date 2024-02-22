"""
实现web界面

>>> gradio unit/app.py demo
"""
from pathlib import Path

import gradio as gr

from unit.detect import detect
from unit.utils import (
    Log,
    find_model_class_using_name,
    get_all_weights,
    get_all_datasets,
)

APP_INTRODUCE = """
# 🖼️风格转换
- 功能：上传本地文件，选择模型、转换风格
- 注意：切换模型后才能使用对应的风格
"""

log = Log("app")
model_name_list = ["cycle_gan", "munit", "funit"]
default_model = model_name_list[0]
default_style = get_all_weights(default_model)[0]
model = None
example_imgs_path = Path.cwd().joinpath("unit/demo")
if example_imgs_path.exists():
    example_imgs = [i for i in example_imgs_path.iterdir() if i.is_file()]
    # FUNIT需要
    class_imgs = [
        i for i in example_imgs_path.joinpath("meerkat").iterdir() if i.is_file()
    ]
else:
    example_imgs = []
    class_imgs = []


def first_load_model():
    global model
    if model is None:
        print("First load model,please wait......")
        model = find_model_class_using_name(default_model)(mode="test")
        print(f"Model [{type(model).__name__}] was created")


def gr_img_mode(mode):
    # 需要修改源代码 source="upload"
    return (
        gr.Image.update(source="webcam")
        if mode == "摄像头"
        else gr.update(source="upload")
    )


def gr_weight_list(model_name):
    model_class = find_model_class_using_name(model_name)
    global model
    if model_class.__name__ != type(model).__name__:
        model = model_class(mode="test")
        log.info(f"Model [{type(model).__name__}] was created")
    else:
        log.warning(f"Model [{type(model).__name__}] was existed")
    gr_model_name = gr.Dropdown.update(
        value=get_all_weights(model_name)[0],
        choices=get_all_weights(model_name),
        label="转换风格",
    )
    if model_name == "funit":
        return gr_model_name  # , gr.update(visible=True)
    else:
        return gr_model_name  # , gr.update(visible=False)


def multi_style_detect(img, model_name, style=None):
    if style is None:
        style = get_all_weights(model_name)
    if img is None:
        assert gr.Error("未上传图片！！！")
    else:
        first_load_model()
        return detect(img, style, model=model)


def single_detect(img, style=None):
    if style is None:
        style = get_all_weights(model_name)
    if img is None:
        assert gr.Error("未上传图片！！！")
    else:
        first_load_model()
        return detect(img, style, model=model)[0]


refresh_symbol = "🔄"


# class FormComponent:
#     def get_expected_parent(self):
#         return gr.components.Form


# class ToolButton(FormComponent, gr.Button):
#     """Small button with single emoji as text, fits inside gradio forms"""

#     def __init__(self, *args, **kwargs):
#         classes = kwargs.pop("elem_classes", [])
#         super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

#     def get_block_name(self):
#         return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args
        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = gr.Button(label="刷新", value=refresh_symbol).style(full_width=True)
    refresh_button.click(fn=refresh, inputs=[], outputs=[refresh_component])
    return refresh_button


def gr_refresh(get_list=get_all_datasets, fun_args=None):
    with gr.Row():
        if fun_args is None:
            with gr.Column(min_width=480):
                choice_datasets = gr.Dropdown(
                    choices=get_list(), value="test", label="选择训练数据"
                )
            create_refresh_button(
                choice_datasets, print, lambda: {"choices": get_list()}
            )
        else:
            with gr.Column(min_width=480):
                choice_datasets = gr.Dropdown(
                    choices=get_list(fun_args), value="test", label="选择训练数据"
                )
            create_refresh_button(
                choice_datasets, print, lambda: {"choices": get_list(fun_args)}
            )
        gr.Markdown("")
    return choice_datasets


def train(dataname):
    from unit.trainer import cycle_gan

    cycle_gan.main(dataname)
    return "训练完成"


CSS_PATH = "utils/statics/style.css"
if Path(CSS_PATH).exists():
    with open(CSS_PATH, "r") as f:
        CSS = f.read()
else:
    CSS = None

demo = gr.Blocks(title="风格转换", css=CSS)
with demo:
    gr.Markdown(APP_INTRODUCE)
    # header
    with gr.Row():
        with gr.Column(scale=2):
            model_name = gr.Dropdown(
                choices=model_name_list, value=model_name_list[0], label="模型选择"
            )
    with gr.Tab(label="单图转换"):
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="选择需要进行风格转换的图片")
            with gr.Column(scale=1):
                img_mode = gr.Radio(["图片", "摄像头"], value="图片", label="上传图片/通过摄像头读取图片")
                img_mode.change(fn=gr_img_mode, inputs=img_mode, outputs=img)
                style = gr.Dropdown(
                    choices=get_all_weights(default_model),
                    value=default_style,
                    label="选择转换风格",
                )
                # class_img_folder_dir = gr.Files(
                #     visible=False,
                #     value=class_imgs,
                #     label="上传图片，funit会向着该图进行风格转换",
                # )
                detect_btn = gr.Button("♻️ 风格转换")
            with gr.Column(scale=1):
                out_img = gr.Image(label="风格图").style(height=256, width=256)
        detect_btn.click(fn=single_detect, inputs=[img, style], outputs=[out_img])
        if example_imgs:
            gr.Examples(example_imgs, inputs=[img], label="示例图片")
    # 2
    with gr.Tab(label="单图多风格预览"):
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="选择需要进行风格转换的图片")
            with gr.Column(scale=1):
                img_mode = gr.Radio(["图片", "摄像头"], value="图片", label="上传图片/通过摄像头读取图片")
                img_mode.change(fn=gr_img_mode, inputs=img_mode, outputs=img)
                btn = gr.Button("♻️ 风格转换(会将所有风格推理一遍)")
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="风格图", elem_id="gallery").style(
                    columns=5, height="auto"
                )
        btn.click(fn=multi_style_detect, inputs=[img, model_name], outputs=gallery)
        if example_imgs:
            gr.Examples(example_imgs, inputs=[img], label="示例图片")
    with gr.Tab(label="训练CycleGAN"):
        choice_datasets = gr_refresh()

        btn_train = gr.Button("📖训练")
        btn_train.click(
            train, choice_datasets, gr.Label(label="训练进度"), show_progress=True
        )
        # xx = gr.Textbox()
        # yy = gr.Textbox()
        # btn_train.click(my_function, None, None, show_progress=True)
    # with gr.Accordion("更多"):
    # gr.Markdown("基于CycleGAN的图像风格转换")
    model_name.change(gr_weight_list, model_name, [style])


def start(share):
    demo.queue(concurrency_count=1)
    demo.launch(
        share=share,
        inbrowser=True,
    )
