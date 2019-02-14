# LeNet-5 TensorFlow Implementation
A TensorFlow implementation of LeNet-5 Model using low-level TensorFlow APIs. The TensorFlow code structures follows 
Andrew NG's suggestions in CS230 Deep Learning 

## Structure of the code
1. **Model is defined under the folder of model/**
* model/input_fn.py: define input data pipeline
* model/model_fn.py: define lenet-5 model
* model/train_fn.py: define how to train the model
* model/eval_fn.py: define how to evaluate the model
* model/utils.py: define some utility functions

**2. Model files (weights) are saved under the folder of experiments/**

**3. Main files for training and evaluating the model**