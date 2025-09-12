import argparse
import json
import signal

from process.tree.interface.interface import Worker

from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QCoreApplication

def progressUpdated(info):

    print(info["progress"]*100)

def onProcessFinished(info):
    print("Process finished")
    QCoreApplication.quit()
    print("Process finished")

def run_cli(args):
    print("Running Tree Eyed in CLI mode...")

    config_filepath = args.config 

    if not config_filepath:
        print("Error: --config is required in CLI mode.")
        return

    # Load the JSON file
    parameters = json.load(args.config)
    #print(parameters)

    app = QCoreApplication([])
    worker = Worker(parameters)

    # Save original signal handlers
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_signal(signum, frame):
        worker.requestInterruption()
        # Restore original handlers so subsequent signals use default behavior
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    worker.finished.connect(onProcessFinished)
    worker.progressUpdated.connect(progressUpdated)
    try:
        worker.start()
        app.exec()
    except Exception as e:
        print(f"Exception occurred: {e}")
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        QCoreApplication.quit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the application in GUI or CLI mode.")
    #parser.add_argument("--cli", action="store_true", help="Run in CLI mode.")
    parser.add_argument(
        '-c', '--config',
        type=argparse.FileType('r'),
        #required=True,
        help='Path to the JSON configuration file'
    )
    parser.add_argument('--parameters', type=argparse.FileType('r'), help='Path to the JSON configuration file')
    # add argument for a json file
    

    args = parser.parse_args()

    run_cli(args)

    # if args.cli:
    #     if not args.task:
    #         args.task = "tiling_detection"
    #     if not args.input:
    #         print("Error: --input is required in CLI mode.")
    #         sys.exit(1)
    #     elif not args.output:
    #         print("Error: --output is required in CLI mode.")
    #         sys.exit(1)
    #     run_cli(args)
    #else:
    #    run_gui()