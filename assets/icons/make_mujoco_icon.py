from pathlib import Path

from PIL import Image

root = Path.home() / "workspace" / "physical-ai-lab"
src = root / "assets" / "icons" / "mujoco-banner.png"
dst = root / "assets" / "icons" / "mujoco.ico"

image = Image.open(src).convert("RGBA")

# 公式バナー左寄りの「M」を正方形で切り出す
width, height = image.size
crop_size = height

left = int(width * 0.18)
top = 0
right = left + crop_size
bottom = height

icon = image.crop((left, top, right, bottom))
icon.save(
    dst,
    format="ICO",
    sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
)

print(dst)
