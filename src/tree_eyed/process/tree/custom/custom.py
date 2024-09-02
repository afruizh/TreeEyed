import torch
import torchvision

import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.transforms import v2 as T

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from ..utils.utils_custom import *

from qgis.core import QgsMessageLog
from qgis.core import Qgis

import os

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class MaskRCNNTreeInference():    
    
    def __init__(self, parameters, models_dir, path_img):
        
        self.parameters = parameters
        self.models_dir = models_dir
        self.path_img = path_img
        
        self.filepath_model = os.path.join(self.models_dir, "MASKRCNNModel.pth")
        self.output_files = []
        
        self.output_dir = parameters["output_path"]
        self.output_prefix = parameters["prefix"]
        self.output_filename = os.path.join(self.output_dir, self.output_prefix + "_vector.shp")
        
        
        self.initialize()
        
    def initialize(self):

        print("initialize")

        # train on the GPU or on the CPU, if a GPU is not available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Force training on CPU
        # device = torch.device('cpu')

        # our dataset has two classes only - background and person
        self.num_classes = 2

        # get the model using our helper function
        self.model = get_model_instance_segmentation(self.num_classes)

        # Load model
        state_dict = torch.load(self.filepath_model, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # move model to the right device
        self.model.to(self.device)

        self.initialized = True

        return
    
    def predict(self, np_image, extent = None, epsg=''):

        model = self.model
        device = self.device

        image = torch.tensor(np_image)
        image = image.permute(2,0,1)

        #image = read_image(input_filepath)
        #eval_transform = model.transform
        eval_transform = get_transform(train=False)
        
        
        QgsMessageLog.logMessage("50%" ,'Tree Eyed Plugin', Qgis.Info) 

        model.eval()
        with torch.no_grad():
            x = eval_transform(image)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]
            
        QgsMessageLog.logMessage("70%" ,'Tree Eyed Plugin', Qgis.Info) 

        result_image = torch.from_numpy(np.zeros((3,image.shape[1], image.shape[2])).astype(np.uint8))
        result_image = result_image[:3, ...]

        valid = 0.6
        masks = (pred["masks"] > valid).squeeze(1)
        #result_image = draw_segmentation_masks(result_image, masks, alpha=1.0, colors="white")

        # Semantic segmentation
        result_image = draw_segmentation_masks(image, masks, alpha=0.35, colors="yellow")
        #result_image = draw_segmentation_masks(image, masks, alpha=1.0, colors="cyan")
        #result_image = draw_segmentation_masks(result_image, masks, alpha=1.0, colors="white")


        result_image = result_image.permute(1, 2, 0)
        result_image = result_image.numpy()

        

        # Obtain as numpy array and transform to tif    
        #utils_custom.np2tif(result_image[0].numpy(), input_filepath_tif, output_filepath)

        

        # #Test save as coco
        #self.save_coco_results([input_filepath_tif], [[pred]], 'D:/local_mydev/test_detectree/experiments/visualization/annotations/test_results.json' )
    
        # # Save as shapefile
        # vis = visualizer.Visualizer()
        # vis.loadCOCO("D:/local_mydata/tree/results/vector/", image)
        # vis.printResults()

        # vis.gdf_tree_bb.to_file(DEFAULT_TEMP_OUTPUT_SHP)

        # Create binary mask
        result_mask =np.zeros((np_image.shape[0], np_image.shape[1],np_image.shape[2])).astype(np.uint8)
        result_mask = torch.tensor(result_mask)
        result_mask = result_mask.permute(2,0,1)
        #result_mask = torch.from_numpy(np.zeros((3,image.shape[1], image.shape[2])).astype(np.uint8))
        #result_mask = result_image[:3, ...]

        # Semantic segmentation
        result_mask = draw_segmentation_masks(result_mask, masks, alpha=1.0, colors="white")

        result_mask = result_mask.permute(1, 2, 0)
        result_mask = result_mask.numpy()


        if extent is not None:

            print("here")
            # self.save_shapefile(pred, extent
            #                     , result_image.shape[1]
            #                     , result_image.shape[0]
            #                     , epsg
            #                     )

            #Save raster binary
            raster_filename = self.output_filename.replace("_vector.shp", "_raster_binary.tif")

            np2tif_extent(result_mask, extent, epsg, raster_filename)

            if not raster_filename in self.output_files:
                self.output_files.append(raster_filename)


            # # Save vector
            # self.save_shapefile_polygon(pred, extent
            #                     , result_image.shape[1]
            #                     , result_image.shape[0]
            #                     , epsg
            #                     )

        return result_mask
        #return result_image
    

        
        