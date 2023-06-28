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
Also, more simply, using the configuration file/data_generation_gaussian.py, you can create data by:

Sampling a Gaussian with random mean and random variance within a compact domain.
Sampling two Gaussians with random mean and random variance and adding them together.
Sampling a Gaussian with random mean and random variance, perturbing it by a polynomial amount, and adding them together.
An example of the execution code would be, e.g.,

```
python3 data_generation.py train_300_test_300 --seed 0 --integration_order 100 --num_train 100 --num_test 100
```

#### Computation of collision integral
The code for learning the operator from the input function f to Q(f, f) using DeepONet is 'train.py'. The code for executing the training using DeepONet is as follows:
```
python3 train.py 3_8_3_8 --model deeponet --seed 0 --gpu 1 --epochs 100000 --lambda 0 --d_t 3 --w_t 8 --d_b 3 --w_b 8 --act tanh --n_basis 8
```
To simultaneously enforce the coefficient of Basis 1 to be 0 and train p+1 basis functions to be orthogonal, you can use the following code:
```
python3 train.py 3_8_3_8_enforce --model deeponet --seed 0 --gpu 2 --epochs 100000 --lambda 0.1 --d_t 3 --w_t 8 --d_b 3 --w_b 8 --act tanh --n_basis 8 --fix_bias
```

### DeepOnet model

DeepONet model is located in the 'model' folder. 'vanila_deeponet.py' is the original model of DeepONet, while 'deeponet.py' is a modified version of the DeepONet model that adds the one additional output of the trunk net instead of the last bias term to suit our collision operator purpose.

### Solution algorithm

#### Linear Boltzmann equation

- [KiT-RT](https://github.com/CSMMLab/KiT-RT)
- [neuralEntropyClosures](https://github.com/ScSteffen/neuralEntropyClosures)
- [Kinetic.jl](https://github.com/vavrines/Kinetic.jl)

#### Nonlinear Boltzmann equation

- [Kinetic.jl](https://github.com/vavrines/Kinetic.jl)