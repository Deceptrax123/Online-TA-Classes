import torch 
from torch import nn 
from Blocks import UnetDecodingBlock,UnetEncodingBlock
from torch.nn import Module,ConvTranspose2d,Conv2d,MaxPool2d,Dropout2d
from torchsummary import summary


class Unet(Module):
    def __init__(self,filters):
        super(Unet,self).__init__()

        #Encoding Blocks
        self.down1=UnetEncodingBlock(filters)
        self.down2=UnetEncodingBlock(2*filters)
        self.down3=UnetEncodingBlock(4*filters)

        #Bottleneck
        self.emb1=Conv2d(in_channels=8*filters,out_channels=8*filters,stride=1,padding=1,kernel_size=3)
        self.emb2=Conv2d(in_channels=8*filters,out_channels=8*filters,stride=1,padding=1,kernel_size=3)

        #Decoding Blocks
        self.up1=UnetDecodingBlock(filters*8)
        self.up2=UnetDecodingBlock(filters*4)
        self.up3=UnetDecodingBlock(filters*2)

        #Maxpool
        self.max1=MaxPool2d(kernel_size=(2,2),stride=2)
        self.max2=MaxPool2d(kernel_size=(2,2),stride=2)
        self.max3=MaxPool2d(kernel_size=(2,2),stride=2)

        #Transpose Convolutions
        self.tconv1=ConvTranspose2d(in_channels=filters*8,out_channels=filters*8,stride=2,padding=1,kernel_size=3,output_padding=1)
        self.tconv2=ConvTranspose2d(in_channels=4*filters,out_channels=4*filters,stride=2,padding=1,kernel_size=3,output_padding=1)
        self.tconv3=ConvTranspose2d(in_channels=2*filters,out_channels=2*filters,stride=2,padding=1,kernel_size=3,output_padding=1)
    
    def forward(self,x):
        x1=self.down1(x) #Input 
        x1_max=self.max1(x1) #Prepared for second encoding block.

        x2=self.down2(x1_max)
        x2_max=self.max2(x2)

        x3=self.down3(x2_max)#256X32X32
        x3_max=self.max3(x3) #256X16X16

        x4=self.emb1(x3_max)
        x5=self.emb2(x4)

        x6=self.tconv1(x5) #output is 256X32X32
        xcat1=torch.add(x6,x3)
        xu1=self.up1(xcat1)

        x7=self.tconv2(xu1)
        xcat2=torch.add(x7,x2)
        xu2=self.up2(xcat2)

        x8=self.tconv3(xu2)
        xcat3=torch.add(x1,x8)
        xu3=self.up(xcat3) #output size same as input size.

        return xu3




