import torch

from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = (256,256) if torch.cuda.is_available() else 128  # use small size if no gpu

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.resize(imsize)
    return image

def show_results(style,content,result):
  fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(24,24))
  ax[0].imshow(content)
  ax[0].title.set_text("Content Image")
  ax[1].imshow(style)
  ax[1].title.set_text("Style Image")
  ax[2].imshow(result)
  ax[2].title.set_text("Resultant Image")
  plt.show()


style_img = image_loader("colors.jpg")                         # Add the paths of the resulting images here
content_img = image_loader("landscape.jpg")
result = image_loader("results/landscape_colors-12.jpg")

show_results(style_img,content_img,result)