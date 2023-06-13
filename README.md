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

#### Computation of collision integral



### DeepOnet model

@TODO

### Solution algorithm

#### Linear Boltzmann equation

- [KiT-RT](https://github.com/CSMMLab/KiT-RT)
- [neuralEntropyClosures](https://github.com/ScSteffen/neuralEntropyClosures)
- [Kinetic.jl](https://github.com/vavrines/Kinetic.jl)

#### Nonlinear Boltzmann equation

- [Kinetic.jl](https://github.com/vavrines/Kinetic.jl)