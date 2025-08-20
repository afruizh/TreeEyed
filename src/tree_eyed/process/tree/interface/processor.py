#from rootprocessor import RootSegmentor

from ..custom_processor import export_coco_dataset
from ..custom_processor import inference


class Processor():

    def __init__(self, params, progress_callback = None, interruption_check = None):
        self.params = params
        self.progress_callback = progress_callback
        self.interruption_check = interruption_check

    def run(self):

        results = self.params


        task = self.params.get("task")

        #***************

        if task == "export_dataset":

            export_coco_dataset(self.params, progress_callback = self.progress_callback, interruption_check = self.interruption_check)
            results.update({"status": "completed", "log": "Task completed succesfully."})

        elif task == "inference":

            inference_results = inference(self.params, progress_callback = self.progress_callback, interruption_check = self.interruption_check)
            #results.update(inference_results)
            results.update({"status": "completed", "log": "Task completed succesfully."})

        # if task == "detection":

        #     input_file = self.params.get("input_file")
        #     output_folder = self.params.get("output_folder")

        #     self.forages_rois_detector = ForagesROIsDetector()
        #     self.forages_rois_detector.inference(input_file, output_folder)



        #     results.update({"status": "completed", "message": "Task completed succesfully."})

        # elif task == "tiling_detection":

        #     input_file = self.params.get("input_file")
        #     output_folder = self.params.get("output_folder")

        #     self.forages_rois_detector = ForagesROIsDetector()
        #     self.forages_rois_detector.tile_inference(input_file, output_folder)

        #     results.update({"status": "completed", "message": "Task completed succesfully."})

        # elif task == "plot_numbering":

        #     input_file = self.params.get("input_file")
        #     output_folder = self.params.get("output_folder")
        #     align = self.params.get("align", False)
        #     serpentine = self.params.get("serpentine", False)

        #     self.forages_rois_detector = ForagesROIsDetector()
        #     self.forages_rois_detector.plot_numbering(input_file, output_folder
        #                                               , align_to_grid=align
        #                                               , serpentine=serpentine)

        #     results.update({"status": "completed", "message": "Task completed succesfully."})

        # elif task == "postprocessing":

        #     input_file = self.params.get("input_file")
        #     output_folder = self.params.get("output_folder")
        #     #if key exists in params, use it, otherwise set default value
        #     align = self.params.get("align", False)
        #     serpentine = self.params.get("serpentine", False)

        #     self.forages_rois_detector = ForagesROIsDetector()
        #     self.forages_rois_detector.plot_numbering(input_file, output_folder
        #                                               , only_postprocess=True
        #                                               , align_to_grid=align
        #                                               , serpentine=serpentine)

        #     results.update({"status": "completed", "message": "Task completed succesfully."})

        # elif task == "tiling_detection_only":

        #     input_file = self.params.get("input_file")
        #     output_folder = self.params.get("output_folder")

        #     self.forages_rois_detector = ForagesROIsDetector()
        #     self.forages_rois_detector.tile_inference(input_file, output_folder, only=True)

        #     results.update({"status": "completed", "message": "Task completed succesfully."})


            

        # if task == "batch_segmentation":

        #     # Perform batch segmentation
        #     input_folder = self.params.get("input_folder")
        #     output_folder = self.params.get("output_folder")

        #     # Initialize the BackgroundRemover and perform batch processing
        #     self.background_remover = RootSegmentor()
        #     self.background_remover.batch_processing(input_folder
        #                                             , output_folder
        #                                             , progress_callback = self.progress_callback
        #                                             , interruption_check = self.interruption_check)
        
        # # elif task == "single_segmentation":
        # #     # Perform batch segmentation
        # #     input_file = self.params.get("input_file")
        # #     output_folder = self.params.get("output_folder")

        # #     # Initialize the BackgroundRemover and perform batch processing
        # #     self.background_remover = BackgroundRemover()
        # #     self.background_remover.inference_file_save(input_file
        # #                                             , output_folder
        # #                                             , progress_callback = self.progress_callback
        # #                                             , interruption_check = self.interruption_check)            


        else:
            results.update({"status": "error", "message": "Invalid task specified."})





        #****************
        print(results)


        return results

    
