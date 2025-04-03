from ImageGenerator import ImageGenerator
from PIL import Image

# load img
img = Image.open("../data/EVA.jpg")

gen = ImageGenerator()
new_img = gen.generate(img, prompt="The huge robot is running",)

new_img.save("output.jpg")