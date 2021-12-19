#importing the necessary libraries
from __future__ import print_function

import torch
import torch.nn as nn

from PIL import Image

import torchvision.transforms as transforms
import torchvision
import torchvision.models as models

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #checking the device configurations
print("device used = ",device)

if torch.cuda.is_available(): torch.cuda.empty_cache()              #empty GPU catche memory

def load_image(image_path, transform=None, shape=None):  #Loads an image and transforms it to tensor
    image = Image.open(image_path).convert('RGB')
    if shape:
        image = image.resize(shape)                      #reshape the given image
    
    if transform:
        image = transform(image).unsqueeze(0)            #add one more dimention for batch size
    
    return image.to(device)                              #returns to device available



class VGGNet(nn.Module):                                 #defining the model class to extract features
  def __init__(self):
    super(VGGNet,self).__init__()
    self.select = ['0','5','10','19','28']                #layer numbers from which the features will be considered
    self.vgg = models.vgg19(pretrained=True).features

  def forward(self, x):
      features = []
      for name, layer in self.vgg._modules.items():
          x = layer(x)
          if name in self.select:
              features.append(x)
      return features


def main(config):
  #defining the transform as used in th original VGG trained on imagenet.
  transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

  im_shape = (256,256)
  style = load_image(config.style_path, transform,shape = im_shape)
  content = load_image(config.content_path, transform, shape= im_shape)

  #setting the initial target variable to be the content image.
  target = content.clone().requires_grad_(True)  #Note: requires_grad is set to true for this variable to train
  #we can set the initial target tot random noise using the following command (uncomment if wanted)
  # target = torch.randn(content.data.size(), device=device).requires_grad_(True)

  vgg = VGGNet().to(device).eval()  #here we are using eval as we dont need our model to train
  print("Model loaded successfully")

  optimizer = torch.optim.LBFGS([target],max_iter=20) #use adam if have low gpu memory as LBFGS takes a lot of memory. Try reducing the image size
                                          #to use LBFGS with low gpu memory

  #uncomment the below code to run adam optimizer and comment the above line
  # optimizer = torch.optim.Adam([target], lr=0.003, betas=[0.5, 0.999]) 

  total_step = 12                       # **IMPORTANT** : If you are using Adam please set the total_step count to be 2000 for similar results to LBFGS 
  style_weight = config.style_weight    #                 Aslo change the log_step size and sample_step accordingly
  log_step = 1                          # logparse while training
  sample_step = 4                       # to save the image

  print("Starting Training...")
  for step in range(total_step):
          
    def closure():
      target_features = vgg(target)
      content_features = vgg(content)
      style_features = vgg(style)
      style_loss = 0
      content_loss = 0
      
      for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss += torch.nn.functional.mse_loss(f2,f1)     # Compute content loss with target and content images

        # Reshape convolutional feature maps
        b, c, h, w = f1.size()             #batch size (b=1)
        f1 = f1.view(b*c, h * w)
        f3 = f3.view(b*c, h * w)

        # Compute gram matrices
        f1 = torch.mm(f1, f1.t())         #matrix multiplication of f1 with its transpose
        f3 = torch.mm(f3, f3.t())

        style_loss += torch.nn.functional.mse_loss(f3,f1) / (c * h * w)   # Compute style loss with target and style images
      
      total_loss = content_loss + style_weight * style_loss      # Compute total loss with appropriate weights
      optimizer.zero_grad()                                # optimization step
      total_loss.backward()

      if (step+1) % log_step == 0:                         #logparse
        print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}, Total Loss: {:.4f}' 
                .format(step+1,total_step, content_loss.item(), style_loss.item(),total_loss.item()))
      
      if (step+1) % sample_step == 0:                      #saveparse
        # Save the generated image
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().squeeze()
        img = denorm(img).clamp_(0, 1)
        torchvision.utils.save_image(img, 'results/sample_result-{}.jpg'.format(step+1))   #can change the directory of saved results here

      return total_loss

    optimizer.step(closure)

  print("Finished Training... check results !!!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--content_path',type=str,default='gal_gadot.jpg')
  parser.add_argument('--style_path',type=str,default='style6.png')
  parser.add_argument('--style_weight',type=int,default=100)
  config = parser.parse_args()
  print(config)
  main(config)





