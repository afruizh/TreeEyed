import os
import json
import webbrowser

from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtCore import QThread

from .processor import Processor

class Worker(QThread):

    finished = Signal(dict)  # Signal emitted when the thread finishes processing
    progressUpdated = Signal(dict) # Add signal for progress (current, total)

    def __init__(self, params):
        super().__init__()
        self.params = params
        # Add a flag for interruption
        self._is_interruption_requested = False

    def run(self):
        """Long-running task."""
        import time
        print("Processing started...")

        processor = Processor(self.params
                              , progress_callback = self.progressUpdated.emit
                              , interruption_check = self.isInterruptionRequested)
        results = processor.run()

        if not self.isInterruptionRequested():
            print("Processing finished!")
            #self.finished.emit({})
            print(results)
            self.finished.emit(results)
        else:
            print("Processing cancelled!")
            # Optionally emit a different signal or specific info for cancellation
            results.update({"status": "cancelled"})
            self.progressUpdated.emit({"status": "Cancelled..."})
            self.finished.emit(results)

    # Override isInterruptionRequested to use our flag
    def isInterruptionRequested(self):
        return self._is_interruption_requested
    
    # Add a method to request interruption
    def requestInterruption(self):
        self._is_interruption_requested = True

class ProcessorInterface(QObject):
    """Interface for processing tasks."""

    msg = Signal(str)
    finished = Signal(dict)
    progressUpdated = Signal(dict) # Relay progress signal

    def initialize(self):
        pass

    @Slot()
    def execute(self):
        print('Execute')

    @Slot()
    def click(self):
        print('click')

    @Slot()
    def download(self):
        print('download')

    @Slot()
    def cancelProcessing(self):
        """Request cancellation of the running worker."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            print("Requesting processing cancellation...")
            self.worker.requestInterruption() # Use the new method

    @Slot(dict)
    def onProcessFinished(self, info):
        """Handle the process completion."""
        self.finished.emit(info)
        print("Process finished signal emitted.")
        # Clean up worker reference
        self.worker = None

    @Slot(str)
    def openOutputFile(self, output_file):
        """Open the output file in Excel."""
        output_file = output_file.replace("file:///","")
        if os.path.exists(output_file):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_file)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(["open", output_file])  # macOS
                    # subprocess.run(["xdg-open", output_file])  # Linux (uncomment if needed)
            except Exception as e:
                print(f"Failed to open file: {e}")
        else:
            print(f"Output file not found: {output_file}")

    @Slot(str)
    def openOutputFolder(self, output_folder):
        """Open the output folder in the default file explorer."""
        if os.path.isdir(output_folder): # Check if it's a valid directory
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder)
                elif os.name == 'posix':  # macOS/Linux
                    if sys.platform == "darwin": # macOS
                        subprocess.run(["open", output_folder])
                    else: # Linux
                        subprocess.run(["xdg-open", output_folder])
                else:
                    print(f"Unsupported OS: {os.name}")
            except Exception as e:
                print(f"Failed to open folder: {e}")
        else:
            print(f"Output folder not found or is not a directory: {output_folder}")

    @Slot(str)
    def open_url(self, url):
        """Open website in default web browser"""
        webbrowser.open(url)

    @Slot(str, str)
    def saveParametersJson(self, file_path, json_data_string):
        """Saves the provided JSON string to the specified file path."""
        try:
            # Parse just to ensure it's valid JSON before writing
            params = json.loads(json_data_string)
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4) # Write with indentation
            print(f"Parameters saved to: {file_path}")
        except Exception as e:
            print(f"Error saving parameters to {file_path}: {e}")
            self.saveLoadError.emit(f"Error saving parameters: {e}")

    @Slot(str)
    def loadParametersJson(self, file_path):
        """Loads parameters from the specified JSON file path."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Parameter file not found: {file_path}")
            with open(file_path, 'r') as f:
                params = json.load(f)
            # Basic validation (optional but recommended)
            if "inputFolder" not in params or "outputFolder" not in params:
                 raise ValueError("Invalid parameter file format.")
            self.parametersLoaded.emit(params) # Emit signal with loaded data
            print(f"Parameters loaded from: {file_path}")
        except Exception as e:
            print(f"Error loading parameters from {file_path}: {e}")
            self.saveLoadError.emit(f"Error loading parameters: {e}")


    @Slot(dict)
    def process(self, params):
        self.worker = Worker(params)
        self.worker.finished.connect(self.onProcessFinished)
        self.worker.progressUpdated.connect(self.progressUpdated)
        self.worker.start()

        