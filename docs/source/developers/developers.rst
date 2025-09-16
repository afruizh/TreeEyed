Developers
============

======================
Command Line Interface
======================

To use TreeEyed functionality in CLI mode you need to create a python environment using:

.. code-block:: bash

	conda env create -f environment.yml
	conda activate tree_eyed

To use GPU processing, make sure cuda and cudnn DLLs are installed in the system or install the following dependencies:

.. code-block:: bash

	conda activate tree_eyed
	conda install conda-forge::cudnn
	conda install conda-forge::libcufft
	conda install conda-forge::cuda-cudart

Additionally, you need to have the models in a local folder and a configuration file, for example a minimal configuration file `example_config.json` is:

.. code-block:: json

	{
		 "model": "HighResCanopyHeight",
		 "model_dir": "path/to/models",
		 "output_path": "path/to/output/folder",
		 "prefix": "output_name",
		 "input_raster_path": "path/to/input/raster",
		 "task": "inference",
		 "raster_outputs": ["grayscale"],
		 "vector_outputs": []
	}

To execute the inference call `tree_eyed_app.py`, for example:

.. code-block:: bash

	python src/tree_eyed/tree_eyed_app.py --config example_config.json

============
Docker Usage
============

You can also use TreeEyed as a docker container.

First build the docker image:

.. code-block:: bash

	docker build -t treeeyed-image .

Run the container interactively:

.. code-block:: bash

	docker run --gpus all -v <path-to-local-workspace-folder>:/app/data -v <path-to-local-models-folder>:/app/models -it treeeyed-image

Execute inference:

.. code-block:: bash

	python src/tree_eyed/tree_eyed_app.py --config /app/data/example_config.json
