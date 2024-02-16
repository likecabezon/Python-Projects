import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


im = Image.open("OutsideTempHistory.gif")
print(im.format, im.size, im.mode)
im = im.convert("RGB")
print(im.format, im.size, im.mode)
im.show()
pixels = im.load()
print(pixels)