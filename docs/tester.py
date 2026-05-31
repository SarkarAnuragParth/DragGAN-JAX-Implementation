from PIL import Image
import numpy as np

image_array = np.ones(shape=(256,256), dtype=np.uint8)

Image.fromarray(image_array, mode='L').save('Mintest.jpg')
