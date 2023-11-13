# CollisionOperatorLearning

DeepONet-based learning of the Boltzmann-type collision operator

## Workflow

### Training set

#### Generation of particle distribution

We use [KiT-RT](https://github.com/CSMMLab/KiT-RT) to produce a bunch of particle distribution functions.
Follow the constructions in [KiT-RT](https://github.com/CSMMLab/KiT-RT) main repository to build the binary file.
Execute the compiled binary by handing over a valid config file, e.g.,

```
KiT-RT config/data_generation_1d.cfg
```
Also, more simply, using the configuration file/data_generation_gaussian.py, you can create toy data by:

Sampling a Gaussian with random mean and random variance within a compact domain.
Sampling two Gaussians with random mean and random variance and adding them together.
Sampling a Gaussian with random mean and random variance, perturbing it by a polynomial amount, and adding them together.
An example of the execution code would be, e.g.,

```
python3 toy_data_generation_gaussian.py train_300_test_300 --seed 0 --integration_order 100 --num_train 100 --num_test 100
```

#### Computation of collision integral
The code for learning the operator from the input function f to Q(f, f) using DeepONet is 'train.py'. The code for executing the training using DeepONet is as follows:
##### 1) vanila DeepONet wo/ bias
```
python3 train.py 1d_3_8_3_8_wo_bias --seed 0 --gpu 0 --dimension 1 --data_file toy --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias no --epochs 100000 --lambda 0
```

##### 2) vanila DeepONet w/ bias
```python3 train.py 1d_3_8_3_8_w_bias --seed 0 --gpu 0 --dimension 1 --data_file toy --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias vanila --epochs 100000 --lambda 0
```

##### 3) (soft constraint) DeepONet with additional orthogonal loss
```
python3 train.py 1d_3_8_3_8_soft_lamb01 --seed 0 --gpu 1 --dimension 1 --data_file toy --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias vanila --epochs 100000 --lambda 0.1
```

##### 4) (Hard constraint) DeepONet with gram schmidt for basis
```
python3 train.py 1d_3_8_3_8_hard_gram --seed 0 --gpu 2 --dimension 1 --data_file toy --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias no --use_gram --epochs 100000 --lambda 0
```

##### 5) (Hard constraint) DeepONet with special bias (depends on input function)
```
python3 train.py 1d_3_8_3_8_hard_special --seed 0 --gpu 3 --dimension 1 --data_file toy --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias depend --epochs 100000 --lambda 0
```

### DeepOnet model

DeepONet model is located in the 'model' folder. 'deeponet.py' is a modified version of the DeepONet model that adds the one additional output of the trunk net instead of the last bias term to suit our collision operator purpose.

### Solution algorithm

#### Linear Boltzmann equation

- [KiT-RT](https://github.com/CSMMLab/KiT-RT)
- [neuralEntropyClosures](https://github.com/ScSteffen/neuralEntropyClosures)
- [Kinetic.jl](https://github.com/vavrines/Kinetic.jl)

#### Nonlinear Boltzmann equation

- [Kinetic.jl](https://github.com/vavrines/Kinetic.jl)