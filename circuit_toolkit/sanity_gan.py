import torch
from circuit_toolkit.GAN_utils import upconvGAN

print("torch", torch.__version__, "cuda available?", torch.cuda.is_available())

G = upconvGAN("fc6").eval()
with torch.inference_mode():
    z = torch.randn(2, 4096)
    imgs = G.visualize(z)  # -> [2, 3, 224, 224], roughly in 0..1

print("OK", imgs.shape, float(imgs.min()), float(imgs.max()))