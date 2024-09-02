import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import numpy as np

import torchmetrics
from torchmetrics.detection import IntersectionOverUnion
import torch
import os
import cv2 as cv

class ResultsDataset():
    def __init__(self, path, result_type = 'coco'):

        self.result_type = result_type # 'coco', 'folder'

        self.coco = None
        self.path = None

        if self.result_type == 'coco':
            self.coco = COCO(path)
            self.path = path            
        elif self.result_type == 'folder':
            self.path = path

        self.imgs = []
        self.imgs = self.get_imgs()

        #List of metrics for semantic segmentation
        self.metrics_ss = []
        self.metrics_ss.append(torchmetrics.Dice(num_classes = 2, ignore_index = 0))
        self.metrics_ss.append(torchmetrics.Precision(task='multiclass', num_classes = 2, ignore_index = 0))
        self.metrics_ss.append(torchmetrics.Recall(task='multiclass', num_classes = 2, ignore_index = 0))
        self.metrics_ss.append(torchmetrics.Accuracy(task='multiclass', num_classes = 2, ignore_index = 0))


    def get_imgs(self):

        if self.result_type == 'folder':
            files = os.listdir(path = self.path)
            basenames = [os.path.basename(f) for f in files] #get only file basename

            images = []
            for index, basename in enumerate(basenames):
                images.append({
                    'id':index
                    ,'file_name':basename
                })

            # {'id': 1,
            #     'width': 256,
            #     'height': 256,
            #     'file_name': 'tile_124.tif',
            #     'license': 0,
            #     'flickr_url': '',
            #     'coco_url': '',
            #     'date_captured': 0}

            return images
        

        return self.coco.loadImgs(self.coco.getImgIds())
    
    def get_pred_img_id(self, filename):
        """
        Returns the img_id corresponding to the same filename
        """ 

        for img in self.imgs:

            file_name = img['file_name']

            if file_name == filename:
                return img['id']

        return -1
    
    def get_ss_mask(self, img_id):

        if img_id == -1:
            return None

        imgs = self.imgs

        # If folder, return image file
        if self.result_type == 'folder':
            filepath = os.path.join(self.path, imgs[img_id]['file_name'])
            mask = cv.imread(filepath)
            mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
            #Convert from [0,255] to [0,1]
            mask = mask / 255
            mask = mask.astype(np.uint8)
            return mask
        
        # else create mask from annotations
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = self.coco.annToMask(anns[0])
        for i in range(len(anns)):
            #mask += vis.coco.annToMask(anns[i])
            mask = np.logical_or(mask, self.coco.annToMask(anns[i]))
        mask = mask.astype(np.uint8)

        return mask
    
    def get_basename(self):
        return os.path.basename(self.path)
    
    def reset_metrics(self):

        for metric in self.metrics_ss:
            metric.reset()

        # for metric in self.metrics_is:
        #     metric.reset()

        # for metric in self.metrics_od:
        #     metric.reset()

class COCOMetrics():
    def __init__(self):
        print('COCOMetrics')

        #List of metrics for semantic segmentation
        self.metrics_ss = []
        self.metrics_ss.append(torchmetrics.Dice(num_classes = 2, ignore_index = 0))
        self.metrics_ss.append(torchmetrics.Precision(task='multiclass', num_classes = 2, ignore_index = 0))
        self.metrics_ss.append(torchmetrics.Recall(task='multiclass', num_classes = 2, ignore_index = 0))
        self.metrics_ss.append(torchmetrics.Accuracy(task='multiclass', num_classes = 2, ignore_index = 0))

        #List of metrics for instance segmentation
        self.metrics_is = []

        #List of metrics for object detection
        self.metrics_od = []
        self.metrics_od.append(IntersectionOverUnion())

        # target values
        self.coco_target = None
        self.coco_target_path = None

        # predicted values
        self.coco_predicted = []
        self.coco_predicted_paths = []

        self.results_target = None
        self.results_preds = []
        
        self.final_message = ""

    def load_target(self, filepath, result_type='coco'):

        #self.coco_target = COCO(filepath)
        #self.coco_target_path = filepath

        self.results_target = ResultsDataset(filepath, result_type=result_type)

        return
    
    def load_pred(self, filepath, result_type='coco'):
        
        #coco = COCO(filepath)
        #self.coco_predicted.append(coco)
        #self.coco_predicted_paths.append(filepath)

        self.results_preds.append(ResultsDataset(filepath, result_type=result_type))

    def load_preds(self, filepaths, result_type='coco'):

        for filepath in filepaths:
            #self.load_pred(filepath)
            self.results_preds.append(ResultsDataset(filepath, result_type=result_type))

    def load(self, filepath_target, filepath_pred, result_type='coco'):

        self.load_target(filepath_target, result_type=result_type)
        self.load_pred(filepath_pred, result_type=result_type)

    def loads(self, filepath_target, filepath_preds, result_type='coco'):
        
        self.load_target(filepath_target, result_type=result_type)
        self.load_preds(filepath_preds, result_type=result_type)

    def compute_metric_final(self, metric):

        metric_res = metric.compute()
        metric_name =  metric.__class__.__name__
        print(f"{metric_name} is: {metric_res}")
        self.final_message +=  f"\n{metric_name} is: {metric_res}"

    def compute_metric_final_list(self, metric_list):

        for metric in metric_list:
            self.compute_metric_final(metric) 


    def compute_metric_step(self, metric, real, pred, id=''):

        metric_res = metric(pred, real)
        metric_name =  metric.__class__.__name__
        print(f"{metric_name} is: {metric_res}")

    def compute_metric_step_list(self, metric_list, real , pred, id=''):
        
        for metric in metric_list:
            self.compute_metric_step(metric, real, pred)
            
    def compute_old(self):

        real_vals = []
        pred_vals = []

        for real, pred in zip(real_vals, pred_vals):

            # Compute for semantic segmentation
            self.compute_metric_step_list(self.metrics_ss, real, pred)
            # Compute for object detection
            self.compute_metric_step_list(self.metrics_is, real, pred)
            # Compute for instance segementation
            self.compute_metric_step_list(self.metrics_od, real, pred)

        # Compute for semantic segmentation
        self.compute_metric_final_list(self.metrics_ss)
        # Compute for object detection
        self.compute_metric_final_list(self.metrics_is)
        # Compute for instance segementation
        self.compute_metric_final_list(self.metrics_od)

    def loadCOCO(self, filepath):

        # Create a COCO object
        self.coco_real = COCO(filepath)


        return
    
    def create_ss_mask(self, coco, img_id):
        """
        Create segmentation mask for image with id img_id
        """       
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            #mask += vis.coco.annToMask(anns[i])
            mask = np.logical_or(mask, coco.annToMask(anns[i]))

        return mask
    
    def extract_bb(self, coco, img_id):
        """
        Extract bounding boxes for img_id
        """       
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        
        bbs  = []
        labels = []

        for ann in anns:
            
            bbox = ann['bbox']
            category_id = ann['category_id']

            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2] 
            ymax = bbox[1] + bbox[3] 

            bb = [xmin, ymin, xmax, ymax]
            #print(bbox)
            #print(bb)
            bbs.append(bb)
            labels.append(category_id)

        pred_bb = {
            "boxes":torch.tensor(bbs)
            , "labels": torch.tensor(labels)
            }

        return [pred_bb]
    
    def get_pred_img_id(self, coco_pred, filename):
        """
        Returns the img_id corresponding to the same filename
        """ 

        img_ids = coco_pred.getImgIds()

        for img_id in img_ids:

            img = coco_pred.loadImgs(img_id)
            file_name = img[0]['file_name']

            if file_name == filename:
                return img_id

        return -1
    
    def reset_metrics(self):

        for metric in self.metrics_ss:
            metric.reset()

        for metric in self.metrics_is:
            metric.reset()

        for metric in self.metrics_od:
            metric.reset()

    def compute(self):

        target_imgs = self.results_target.imgs

        # Iterate over images in target
        for target_img in target_imgs:
            print(target_img)

            file_name = target_img['file_name']

            # Check if preds contains same img
            img_ids_pred = []
            for results_pred in self.results_preds:
                img_ids_pred.append(results_pred.get_pred_img_id(file_name))


            # If at least on contains
            if max(img_ids_pred) > 0:

                # Obtain targets
                img_id = target_img['id']
                mask_target = self.results_target.get_ss_mask(img_id)

                rows = 1
                cols = len(self.results_preds)+1
                count = 1
                fig = plt.figure()
                fig.add_subplot(rows, cols, count)
                plt.imshow(mask_target.astype(np.float32))

                pred_masks = []
                # For each pred obtain preds
                for index, results_pred in enumerate(self.results_preds):
                    pred_id = img_ids_pred[index]
                    mask_pred = results_pred.get_ss_mask(pred_id)
                    pred_masks.append(mask_pred)

                    count = count + 1
                    if mask_pred is not None:                        
                        fig.add_subplot(rows, cols, count)
                        plt.imshow(mask_pred.astype(np.float32))

                #Compute
                for index_pred, mask_pred in enumerate(pred_masks):
                    if mask_pred is not None:
                        print('For file',index_pred,self.results_preds[index_pred].get_basename())
                        # self.compute_metric_step_list(self.metrics_ss
                        #                             , torch.from_numpy(mask_target)
                        #                             , torch.from_numpy(mask_pred))
                        self.compute_metric_step_list(self.results_preds[index_pred].metrics_ss
                                                    , torch.from_numpy(mask_target)
                                                    , torch.from_numpy(mask_pred))
                
        for results_pred in self.results_preds:
            print("****************************")
            print("Final results for pred",index_pred,results_pred.get_basename())
            
            self.final_message =  "****************************"
            self.final_message += "\nFinal results for pred {}".format(results_pred.get_basename())
                     
            self.compute_metric_final_list(results_pred.metrics_ss)
            results_pred.reset_metrics()

        # self.compute_metric_final_list(self.metrics_ss)
        # self.reset_metrics()
