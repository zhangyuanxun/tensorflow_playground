# LeNet-5 TensorFlow Implementation
![LeNet-5 Architecture](lenet5.png)

A TensorFlow implementation of LeNet-5 Model using low-level TensorFlow APIs. The TensorFlow code structures follows 
Andrew NG's suggestions in CS230 Deep Learning.


## Code structure
1. **Model is defined under the folder of model**

* model/input_fn.py: define input data pipeline
* model/model_fn.py: define lenet-5 model
* model/train_fn.py: define how to train the model
* model/eval_fn.py: define how to evaluate the model
* model/utils.py: define some utility functions

2. **Model folder for saving hyper-parameters and weights are defined under the folder of experiments/**
* experiments/base_model: define model directory
* experiments/base_model/params.json: a json file to define hyper-parameters

3. **Main files for training and evaluating the model**
* train.py: main file for training the model
* evaluate.py: main file for evaluating model


## Usage
1. **Train the model**
```
python train.py --model_dir experiments/base_model
```

2. **Evaluate the model**
```
python evaluate.py --model_dir experiments/base_model --restore_from last_weights
```


## Reference
LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324. [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)