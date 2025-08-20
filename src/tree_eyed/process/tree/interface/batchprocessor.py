import os
import pathlib
import glob

class BatchProcessor():

    def __init__(self):
        return
    
    def batch_process(self, input_dir, output_dir
                      , output_suffixes = ["output"]
                      , format="jpg"
                      , pattern='**/*.tiff'
                      , processing_fc=None
                      , output_format = None
                      , progress_callback=None
                      , interruption_check=None
                      ):

        if processing_fc == None:
            print("Processing function is None")
            return
        else:

            logs = []

            if output_format == None:
                output_format = format

            # Get list of files in folder and subfolders
            pattern = '**/*.'  + format
            files = glob.glob(pattern, root_dir=input_dir, recursive=True)
            total_files = len(files)
            processed_count = 0

            # Emit initial progress if needed
            if progress_callback:
                progress_callback({"processed_count":processed_count
                                    , "total_files":total_files
                                    , "status":"Initializing..."
                                    , "logs":logs
                                    , "percent": processed_count/total_files*100
                                    })

            for file in files:

                # Check for interruption request before processing each file
                if interruption_check and interruption_check():
                    print("Interruption requested, stopping batch process.")
                    break # Exit the loop

                filepath = os.path.join(input_dir, file)
                basename = os.path.basename(filepath)
                parent_dir = os.path.dirname(file)  # Fix to use the correct relative path for the file
                output_sub_dir = os.path.join(output_dir, parent_dir)  # Ensure output directory keeps the same structure

                # Create output filepath list
                output_filepaths = []
                for suffix in output_suffixes:
                    output_filepaths.append(os.path.join(output_sub_dir, basename.replace("." + format, "_" + suffix + "." + output_format)))

                if not os.path.exists(output_filepaths[0]):  # Process only if first output file does not exist

                    if not os.path.exists(output_sub_dir):  # Create subfolders if necessary
                        pathlib.Path(output_sub_dir).mkdir(parents=True, exist_ok=True)

                    logs.append(f"Processing {file}")
                    
                    processing_fc(filepath, output_filepaths) # Process and save file

                    print(file)
                    print(output_filepaths[0])
                    print("****")
                    logs.append(f"Saved {output_filepaths[0]}")

                processed_count += 1
                # Emit progress after attempting to process (or skip) each file
                if progress_callback:
                    progress_callback({"processed_count":processed_count
                                       , "total_files":total_files
                                       , "status":"Processing"
                                       , "logs": logs
                                       , "percent": processed_count/total_files*100
                                       })

            print(f"Batch process loop finished. Processed {processed_count}/{total_files} files.")