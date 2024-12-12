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

def find_and_remove_solid_shapes(binary_image, solidity_threshold=0.9):
    # Find contours of all shapes
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for solid shapes
    solid_shapes_mask = np.zeros_like(binary_image)
    
    for cnt in contours:
        # Compute contour area and convex hull area
        area = cv.contourArea(cnt)
        if area > 0:  # Avoid division by zero
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                
                # Check if the shape meets the solidity threshold
                if solidity >= solidity_threshold:
                    cv.drawContours(solid_shapes_mask, [cnt], -1, 255, thickness=cv.FILLED)
    
    # Remove solid shapes from the binary image
    remaining_shapes = cv.bitwise_and(binary_image, cv.bitwise_not(solid_shapes_mask))
    
    return solid_shapes_mask, remaining_shapes

def iterative_closing(binary_image, kernel_size=(5, 5)):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    final_result = np.zeros_like(binary_image)  # To store all full shapes over iterations

    remaining_shapes = binary_image.copy()

    max_count = 100
    count = 0
    
    while (count < max_count and np.any(remaining_shapes)):  # Stop when the image is completely black
        # Find and remove full shapes
        full_shapes_mask, remaining_shapes = find_and_remove_solid_shapes(remaining_shapes)
        
        # Add full shapes to the final result
        final_result = cv.bitwise_or(final_result, full_shapes_mask)
        
        # Close the remaining shapes
        remaining_shapes = cv.morphologyEx(remaining_shapes, cv.MORPH_CLOSE, kernel)

        count = count + 1
        print(count)
    
    return final_result

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


        # Post process
        # Correct shapes
        # Apply the iterative closing process
        result_mask = cv.cvtColor(result_mask, cv.COLOR_RGB2GRAY)
        kernel_size = (8,8)  # Adjust based on the image
        result_mask = iterative_closing(result_mask, kernel_size)
        result_mask = cv.cvtColor(result_mask, cv.COLOR_GRAY2RGB)


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
    

        
        