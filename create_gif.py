from PIL import Image
import glob

frames = []
img = glob.glob("samples_8x8/*.png")
for i in img:
    frames.append(Image.open(i))
frames[0].save("gif_training.gif",
               format="GIF",
               append_images=frames,
               save_all=True,
               duration=300,
               loop=0)