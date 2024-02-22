from unit.detect import detect
from unit.utils.scorce import calc_score_from_dir
from pathlib import Path
from unit.models import CycleGANModel

img1_dir = "data/scenery2waterink/trainA"
img2_dir = "logs/images/scenery2waterink"

# 推理出所有图片
img1_paths = Path(img1_dir).iterdir()
model = CycleGANModel(mode="test")
for i in img1_paths:
    detect(
        image=str(i),
        styles="horse2waterink.pth",
        model=model,
        save_image=True,
    )

# 计算指标
# calc_score_from_dir(img1_dir, img2_dir, 'SSIM')
# calc_score_from_dir(img1_dir, img2_dir, 'PSNR')
# calc_score_from_dir(img1_dir, img2_my_dir, 'SSIM')
# calc_score_from_dir(img1_dir, img2_my_dir, 'PSNR')
