# Dependencies

# Structure
This project follows the setup from:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide2/Research_Projects.html

general/
│   train.py
│   task.py
│   mutils.py
layers/
experiments/
│   task1/
│        train.py
│        task.py
│        eval.py
│        dataset.py
│   task2/
│        train.py
│        task.py
│        eval.py
│        dataset.py

The general/train.py file summarizes the default operations every model needs (training loop, loading/saving model, setting up model, etc.). If you use PyTorch Lightning, this reduces to a train file per task, and only needs the specification of the trainer object.

The general/task.py file summarizes a template for the specific parts you have to do for a task (training step, validation step, etc.). If you use PyTorch Lightning, this would be the definition of the Lightning Module.

The layers/models folder contains the code for specifying the nn.Modules you use for setting up the model.

The experiments folder contains the task-specific code. Each task has its own train.py for specifying the argument parser, setting up the model, etc., while the task.py overwrites the template in general/task.py. The eval.py file should have as input a checkpoint directory of a trained model, and should evaluate this model on the test dataset. Finally, the file dataset.py contains all parts you need for setting up the dataset.

# Examples

`python -m experiments.gbif_jax.train --help`

