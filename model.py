import torch
import torch.nn as nn
from torchsummary import summary
import albumentations as A
from torchvision.transforms import CenterCrop

def CNN_Block_decoder(input,output):

    Block = nn.Sequential(
          nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=2, stride = 2),
          nn.Conv2d(in_channels=output,out_channels=output,kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(in_channels=output,out_channels=output,kernel_size=3),
          nn.ReLU()
        )
    return Block

def CNN_Block_encoder(input,output):
    Block = nn.Sequential(
          nn.Conv2d(in_channels=input,out_channels=output,kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(in_channels=output,out_channels=output,kernel_size=3),
          nn.ReLU()
        )
    return Block




class Unet(nn.Module):
    def __init__(self,input_channel,output_channel): # output chanel is the number of classes wanted of segmentation
        super(Unet, self).__init__()
        self.block_down1 = CNN_Block_encoder(input_channel, 64)
        self.block_down2 = CNN_Block_encoder(64, 128)
        self.block_down3 = CNN_Block_encoder(128, 256)
        self.block_down4 = CNN_Block_encoder(256, 512)
        self.bottom = CNN_Block_encoder(512, 1024)

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride=2)



        self.upsampling1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride = 2)
        self.upsampling2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride = 2)
        self.upsampling3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride = 2)
        self.upsampling4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride = 2)

        self.block_upCNN1 = CNN_Block_encoder(128, 64)
        self.block_upCNN2 = CNN_Block_encoder(256, 128)
        self.block_upCNN3 = CNN_Block_encoder(512, 256)
        self.block_upCNN4 = CNN_Block_encoder(1024, 512)

        self.final_layer = nn.Conv2d(in_channels=64,out_channels=output_channel,kernel_size=1)


    def CenterCrop_cat(self,x,feature):
        if x.shape != feature.shape:
            (_,_,h,w) = x.shape
            features_cropped = CenterCrop([h, w])(feature)
        return torch.cat((features_cropped,x),dim=1)  # dim=1 because concat on the channel layer



    def forward(self,input):
        skip_connection = [] # we will store the intermediate values of encoder here, they will feed the decoder next.
        x = self.block_down1(input)
        skip_connection.append(x)
        x = self.Maxpool(x)

        x = self.block_down2(x)
        skip_connection.append(x)
        x = self.Maxpool(x)


        x = self.block_down3(x)
        skip_connection.append(x)
        x = self.Maxpool(x)


        x = self.block_down4(x)
        skip_connection.append(x)
        x = self.Maxpool(x)



        #print(f'skip connection : {skip_connection[-1].shape}')

        x = self.bottom(x) #The output of convTranspose2d is o = (i -1)*s - 2*p + f + output_padding => (28 -1)*2 - 2 * 0 +2 +0 =  56


        #print(f'bottom = {x.shape}')
        x = self.upsampling1(x)
        #print(f'upsampling = {x.shape}')
        x = self.CenterCrop_cat(x=x,feature = skip_connection[-1])
        #print(f"concat = {x.shape}")
        x = self.block_upCNN4(x)


        x = self.upsampling2(x)
        x = self.CenterCrop_cat(x=x,feature = skip_connection[-2])
        x = self.block_upCNN3(x)
        
        x = self.upsampling3(x)
        x = self.CenterCrop_cat(x=x,feature = skip_connection[-3])
        x = self.block_upCNN2(x)
        
        x = self.upsampling4(x)
        x = self.CenterCrop_cat(x=x,feature = skip_connection[-4])
        x = self.block_upCNN1(x)

        x = self.final_layer(x)


        return x



def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model= Unet(input_channel=1, output_channel=1).to(device)
    x = torch.randn((7,1,572,572)).to(device)  #batch_size,nb_channel,width,height
    preds = model(x)
    print(preds.shape)

    #summary(model,(1,572,572))


test()


