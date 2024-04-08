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