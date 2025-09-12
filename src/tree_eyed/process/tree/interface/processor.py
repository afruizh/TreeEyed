#from rootprocessor import RootSegmentor

from ..custom_processor import export_coco_dataset
from ..custom_processor import inference
from ..custom_processor import raster2vector
from ..custom_processor import filter_area


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

            task_results = inference(self.params, progress_callback = self.progress_callback, interruption_check = self.interruption_check)
            #results.update(inference_results)
            if task_results is not None and "status" in task_results and "log" in task_results:
                results.update({"status": task_results["status"], "log": task_results["log"]})
            else:
                results.update({"status": "completed", "log": "Task completed succesfully."})

        elif task == "raster2vector":

            task_results = raster2vector(self.params, progress_callback = self.progress_callback, interruption_check = self.interruption_check)
            if task_results is not None and "status" in task_results and "log" in task_results:
                results.update({"status": task_results["status"], "log": task_results["log"]})
            else:
                results.update({"status": "completed", "log": "Task completed succesfully."})

        elif task == 'filter_area':

            task_results = filter_area(self.params, progress_callback = self.progress_callback, interruption_check = self.interruption_check)
            if task_results is not None and "status" in task_results and "log" in task_results:
                results.update({"status": task_results["status"], "log": task_results["log"]})
            else:
                results.update({"status": "completed", "log": "Task completed succesfully."})



        else:
            results.update({"status": "error", "message": "Invalid task specified."})





        #****************
        print(results)


        return results

    
