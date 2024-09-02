import argparse
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchmetrics
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import math
import torchvision.transforms.functional as TF
import torchvision
from torchvision.utils import save_image

from .models.backbone import SSLVisionTransformer
from .models.dpt_head import DPTHead
import pytorch_lightning as pl
#from .models.regressor import RNet


import cv2 as cv
import rasterio as rio

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from qgis.core import Qgis
from qgis.core import QgsMessageLog

def np2tif_2(data, filepath_tif, filepath_output, output_dtype=rio.uint8):

    # Load original tif file and copy metadata
    orig_img = rio.open(filepath_tif)
    out_meta = orig_img.meta.copy()
    out_meta.update({'count':1},indexes=1)

    # Save file
    with rio.open(filepath_output, "w", **out_meta) as dst:
        dst.write(data.astype(output_dtype))

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

class SSLAE(nn.Module):
    def __init__(self, pretrained=None, classify=True, n_bins=256, huge=False):
        super().__init__()
        if huge == True:
            self.backbone = SSLVisionTransformer(
            embed_dim=1280,
            num_heads=20,
            out_indices=(9, 16, 22, 29),
            depth=32,
            pretrained=pretrained
            )
            self.decode_head = DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )  
        else:
            self.backbone = SSLVisionTransformer(pretrained=pretrained)
            self.decode_head = DPTHead(classify=classify,n_bins=256)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x) 
        return x

class SSLModule(pl.LightningModule):
    def __init__(self, 
                  ssl_path="compressed_SSLbaseline.pth"):
        super().__init__()
    
        if 'huge' in ssl_path:
            self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        else:
            self.chm_module_ = SSLAE(classify=True, huge=False).eval()
        
        if 'compressed' in ssl_path:   
            ckpt = torch.load(ssl_path, map_location='cpu')
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_, 
                {torch.nn.Linear,torch.nn.Conv2d,  torch.nn.ConvTranspose2d},
                dtype=torch.qint8)
            self.chm_module_.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ssl_path)
            state_dict = ckpt['state_dict']
            self.chm_module_.load_state_dict(state_dict)
        
        self.chm_module = lambda x: 10*self.chm_module_(x)
    def forward(self, x):
        x = self.chm_module(x)
        return x

class NeonDataset(torch.utils.data.Dataset):
    # path = './data/images/'
    # root_dir = Path(path)
    # df_path = './data/neon_test_data.csv'
    #path = os.path.join(self.base_path, "data/images")
    #root_dir = Path(path)
    #df_path = os.path.join(base_path, 'data/neon_test_data.csv')
    
    def __init__(self, model_norm, new_norm, path_img = '', src_img='maxar', 
                 trained_rgb= False, no_norm = False,
                **kwargs):
       
        self.no_norm = no_norm
        self.model_norm = model_norm
        self.new_norm = new_norm
        self.trained_rgb = trained_rgb
        self.size = 256
        #self.df = pd.read_csv(self.df_path, index_col=0)
        self.src_img = src_img
        self.path_img = path_img

        print(self.path_img)

        img = cv.imread(self.path_img)
        (self.h, self.w, channels) = img.shape

        print('w',self.w)
        print('h',self.h)

        self.cols = int(np.ceil(self.h/self.size))
        self.rows = int(np.ceil(self.w/self.size))
        
        # number of times crops can be used horizontally
        self.size_multiplier = 6 
        
    def __len__(self):
        # if self.src_img == 'neon':
        #     return 30 * len(self.df) 
        # return len(self.df)
        #return 1
        return int(self.cols*self.rows)
        

    def __getitem__(self, i):
        print("__getitem__")    

        total = self.cols*self.rows


        jx = i // self.cols  # Get the row number based on the index
        jy = i % self.cols  # Get the column number based on the index

        x = list(range(0, self.w, self.size))
        x = x[jx]
        
        y = list(range(0, self.h, self.size))
        y = y[jy]
        
        print("jx jy", jx,jy)
        print("x y", x,y)

        name = self.path_img
        img = TF.to_tensor(Image.open(name).crop((x, y, x+self.size, y+self.size)))

    
        return {'img': img, 
                    'img_no_norm': img, 
                    'chm': img,
                    #'lat':torch.Tensor([l.lat]).nan_to_num(0),
                    #'lon':torch.Tensor([l.lon]).nan_to_num(0),
                    'jx':jx,
                    'jy':jy,
                    'x':x,
                    'y':y
                }

class HRCHInference():

    def __init__(self, parameters, models_dir, path_img):

        self.parameters = parameters

        #self.base_path = r"D:\local_mydata\models\trees\HighResCanopyHeight"
        
        self.models_dir = models_dir
        #self.temp_dir = temp_dir
        
        #self.base_path = HRCH_PATH

        
        self.base_path = models_dir
        print("self.base_path")
        print(self.base_path)

        self.model = []
        self.norm = []
        self.model_norm = []
        self.device = "cpu"
        #self.device = "cuda:0"
        #self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.img_result_binary = None
        self.hrch_type = parameters["hrch_type"]

        self.output_files = []

        if (self.hrch_type == "Satellite"):
            #self.args_checkpoint = 'saved_checkpoints/compressed_SSLlarge.pth'
            self.args_checkpoint = 'compressed_SSLlarge.pth'
        elif (self.hrch_type == "Aerial"):
            #self.args_checkpoint = 'saved_checkpoints/compressed_SSLhuge_aerial.pth'
            self.args_checkpoint = 'compressed_SSLhuge_aerial.pth'
        else:
            #self.args_checkpoint = 'saved_checkpoints/compressed_SSLlarge.pth'
            self.args_checkpoint = 'compressed_SSLlarge.pth'

        print("Using " + self.args_checkpoint)


        self.path_img = path_img


        #self.path_img_output = "D:/local_mydata/tree/results/vector/result_raster_output.tif"

        self.path_img_output = parameters["output_path"] + "/" + parameters["prefix"] + "_raster.tif"
        
        self.hrch_threshold = parameters["hrch_threshold"]

        self.initialize()

        return
    
    def initialize(self):

        # # 1- load network and its weight to normalize aerial images to match intensities from satellite images. 
        # norm_path = os.path.join(self.base_path, 'saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt')

        # ckpt = torch.load(norm_path, map_location='cpu')

        # state_dict = ckpt['state_dict']
        # for k in list(state_dict.keys()):
        #     if 'backbone.' in k:
        #         new_k = k.replace('backbone.','')
        #         state_dict[new_k] = state_dict.pop(k)

        # self.model_norm = RNet(n_classes=6)
        # self.model_norm = self.model_norm.eval()
        # self.model_norm.load_state_dict(state_dict)

        # 2- load SSL model
        print("LOAD SSL MODEL")
        print(self.base_path)
        print(os.path.join(self.base_path, self.args_checkpoint))
        os.path.join(self.base_path, self.args_checkpoint)
        self.model = SSLModule(ssl_path = os.path.join(self.base_path, self.args_checkpoint))
        self.model.to(self.device)
        self.model = self.model.eval()

        # 3- image normalization for each image going through the encoder
        self.norm = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
        self.norm = self.norm.to(self.device)

    def evaluate2(self, model, 
             norm, 
             model_norm,
             name, 
             bs=32, 
             trained_rgb=False,
             normtype=2,
             device = 'cuda:0', 
             no_norm = False, 
             display = False):
      
        dataset_key = 'neon_aerial'
        
        print("normtype", normtype)    
        
        # choice of the normalization of aerial images. 
        # i- For inference on satellite images args.normtype should be set to 0; 
        # ii- For inference on aerial images, if corresponding Maxar quantiles at the
        # same coordinates are known, args.normtype should be set to 1;
        # iii- For inference on aerial images, an automatic normalization using a pretrained
        # network on aerial and satellite images on Neon can be used: args.normtype should be set to 2 (default); 
        
        new_norm=True
        no_norm=False
        if normtype == 0:
            no_norm=True
        elif normtype == 1:
            new_norm=False
        elif normtype == 2:
            new_norm=True
        
        ds = NeonDataset( model_norm, new_norm, path_img = self.path_img, domain='test', src_img='neon', trained_rgb=trained_rgb, no_norm=no_norm)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
            
        #Path('../reports').joinpath(name).mkdir(parents=True, exist_ok=True)
        #Path('../reports/'+name).joinpath('results_for_fig_'+dataset_key).mkdir(parents=True, exist_ok=True)
        path_reports = os.path.join(self.base_path, 'reports')
        #Path(path_reports).joinpath(name).mkdir(parents=True, exist_ok=True)
        #Path(os.path.join(path_reports,name)).joinpath('results_for_fig_'+dataset_key).mkdir(parents=True, exist_ok=True)
        
        preds= []
        
        fig_batch_ind = 0

        final_img = np.zeros((ds.h, ds.w))
        final_img_float = np.zeros((ds.h, ds.w))

        #for batch in tqdm(dataloader):
        for batch in dataloader:

            QgsMessageLog.logMessage("15%" ,'Tree Eyed Plugin', Qgis.Info)     
            print('batch')
            batch = {k:v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = model(norm(batch['img']))
            pred = pred.cpu().detach().relu()
            QgsMessageLog.logMessage("20%" ,'Tree Eyed Plugin', Qgis.Info) 

            print("shape", pred.shape)
            
            if display == True:
                # display Predicted CHM
                for ind in range(pred.shape[0]):
                    
                    progress = ind/pred.shape[0]*70+20
                    QgsMessageLog.logMessage(str(progress) + "%" ,'Tree Eyed Plugin', Qgis.Info)     
                    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
                    #plt.subplots_adjust(hspace=0.5)
                    #img_no_norm = batch['img_no_norm'][ind].cpu()
                    #Inn = np.moveaxis(img_no_norm.numpy(), 0, 2)
                    #img = batch['img'][ind].cpu()
                    #I = np.moveaxis(img.numpy(), 0, 2)
                    # ax[0].imshow(Inn)
                    # ax[0].set_title(f"Image",fontsize=12)
                    # ax[0].set_xlabel('meters')
                    # ax[1].imshow(I)
                    # ax[1].set_title(f"Normalized Image ",fontsize=12)
                    # ax[1].set_xlabel('meters')
                    #combined_data = np.concatenate((batch['chm'][ind].cpu().numpy(), pred[ind].detach().numpy()), axis=0)
                    #_min, _max = np.amin(combined_data), np.amax(combined_data)
                    #pltim = ax[2].imshow(pred[ind][0].detach().numpy(), vmin = _min, vmax = _max)
                    # ax[2].set_title(f"Pred CHM",fontsize=12)
                    # ax[2].set_xlabel('meters')
                    #cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
                    #fig.colorbar(pltim, cax=cax, orientation="vertical")
                    #cax.set_title("meters", fontsize=12)
                    #print("here")
                    #normtype2 = 'aerial_normalization_quantiles_predictor.ckpt'
                    #print(f"{name}/fig_{fig_batch_ind}_{ind}_{normtype2}.png")
                    #save_filename = 
                    #plt.savefig(f"{name}/fig_{fig_batch_ind}_{ind}_{normtype2}.png", dpi=300)

                    # im = Image.fromarray(pred[ind][0].detach().numpy())
                    # if im.mode != 'RGB':
                    #     im = im.convert('RGB')
                    # im.save(f"{name}/fig_{fig_batch_ind}_{ind}_{normtype2}.jpg")

                    cv_im0 = pred[ind][0].detach().numpy()
                    print(np.max(cv_im0))
                    print(np.min(cv_im0))
                    #cv_im = (cv_im > 0.1)*(cv_im < 3)
                    #cv_im = (cv_im0 > 0.1)
                    #cv_im = cv_im*255
                    
                    # Normalization
                    # cv_im = normalize8(cv_im0)
                    # cv_im = (cv_im > 40)*255
                    
                    cv_im = cv_im0

                    print(np.max(cv_im))
                    print(np.min(cv_im))
                    print(cv_im.dtype)

                    mask_binary = cv_im

                    #ret,mask_binary = cv.threshold(cv_im,25,255,cv.THRESH_BINARY)
                    #cv.imwrite(f"{name}/fig_{fig_batch_ind}_{ind}_{normtype2}_CV.jpg", mask_binary)
                    
                    sub = mask_binary
                    #sub_row = sub.shape[0]
                    #sub_col = sub.shape[1]
                    #jx = batch['jx'][ind].cpu().numpy()
                    #jy = batch['jy'][ind].cpu().numpy()
                    x = batch['x'][ind].cpu().numpy()
                    y = batch['y'][ind].cpu().numpy()



                    print("minx",x+256,final_img.shape[0])
                    print("miny",y+256,final_img.shape[1])

                    x2 = min(x+256, final_img.shape[1])
                    y2 = min(y+256, final_img.shape[0])

                    print("x",x,x2)
                    print("y",y,y2)
                    print("xx",0,x2-x)
                    print("yy",0,y2-y)

                    # if (np.max(cv_im0) > 0.05):
                    #     cv_im0 = normalize8(cv_im0)
                    # else:
                    #     cv_im0 = cv_im0*0
                    # print(cv_im0.shape)
                    
                    if not (np.max(cv_im0) > 0.05):
                        cv_im0 = cv_im0*0
                    
                    final_img_float[y:y2, x:x2] = cv_im0[0:y2-y,0:x2-x]
                    #final_img_float[y:y2, x:x2] = cv_im0[0:y2-y,0:x2-x]

                    max_value = np.max(cv_im0)
                    value = self.hrch_threshold*max_value
                    cv_im = (cv_im0 > value)*255
                    final_img[y:y2, x:x2] = cv_im[0:y2-y,0:x2-x]

                    
                    # x2 = min(final_img.shape[0],jx*256+sub_row)
                    # print("x min of", final_img.shape[0], (jx+1)*256+sub_row)
                    # y2 = min(final_img.shape[1],jy*256+sub_col)
                    # print("y min of", final_img.shape[1], (jy+1)*256+sub_col)

                    # print(jy*256,":",y2)
                    # print(jx*256,":",x2)

                    # print("shape",(y2-jy*256,x2-jx*256))
                    
                    #final_img[jx*256:x2, jy*256:y2] = sub[0:x2-jx*256,0:y2-jy*256]

                    #final_img[jy*256:y2, jx*256:x2] = sub[0:y2-jy*256,0:x2-jx*256]
                    #cv_im0 = normalize8(cv_im0)
                    #final_img_float[jy*256:y2, jx*256:x2] = cv_im0[0:y2-jy*256,0:x2-jx*256]
                
                fig_batch_ind = fig_batch_ind + 1

            #preds.append(pred)

            if display:
                break
        #preds = torch.cat(preds)

        # cv.imwrite(f"{name}/final_img_CV.jpg", final_img)
        # print(final_img.shape)
        final_img_2 = np.expand_dims(final_img, axis=0)

        if "binary" in self.parameters["raster_outputs"]:
            np2tif_2(final_img_2, self.path_img, self.path_img_output.replace("_raster","_raster_binary"))

            #add result filepath to list of results
            if not self.path_img_output.replace("_raster","_raster_binary") in self.output_files:
                self.output_files.append(self.path_img_output.replace("_raster","_raster_binary"))

        result_binary = np.squeeze(final_img_2)
        #result_binary = np.moveaxis(result_binary, 0, 1)
        result_binary = result_binary.astype(np.uint8)
        
        self.img_result_binary = result_binary

        # #final_img_float = normalize8(final_img_float)
        final_img_float = np.expand_dims(final_img_float, axis=0)
        # #np2tif_2(final_img_float, self.path_img, self.path_img_output.replace("_output", "_output_float"), output_dtype=rio.float32)
        
        if "grayscale" in self.parameters["raster_outputs"]:
            np2tif_2(final_img_float, self.path_img, self.path_img_output.replace("_output", "_output_float"))
        
            #add result filepath to list of results
            if not self.path_img_output.replace("_output", "_output_float") in self.output_files:
                self.output_files.append(self.path_img_output.replace("_output", "_output_float"))

    def predict(self):

        self.output_files = []

        print("HRCH predict called")
        #return

        # 4- evaluation 

        args_name = os.path.join(self.base_path, 'output_inference')
        args_trained_rgb = False
        #args_normtype = os.path.join(base_path, 'saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt')
        args_normtype = 2
        args_display = True


        self.evaluate2(self.model, self.norm
                , self.model_norm
                , name=args_name
                , bs=100
                , trained_rgb=args_trained_rgb
                , normtype=args_normtype
                , device=self.device
                , display=args_display)

