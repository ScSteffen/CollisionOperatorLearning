#### JaeYong


### (Entropy)
### 1) Entropy 1D isotropic
## vanila DeepONet wo/ bias
python3 ../train_HG.py HG_00_wo_bias --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias no --epochs 10000 --lambda 0

## vanila DeepONet w/ bias
python3 ../train_HG.py HG_00_w_bias --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias vanila --epochs 10000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 ../train_HG.py HG_00_soft_lamb01 --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias vanila_ortho --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 ../train_HG.py HG_00_hard_gram --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias gram --use_gram --epochs 10000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 ../train_HG.py HG_00_hard_special --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias depend --epochs 10000 --lambda 0


### 2) Entropy 1D HG
## vanila DeepONet wo/ bias
python3 ../train_HG.py HG_09_wo_bias --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias no --epochs 10000 --lambda 0

## vanila DeepONet w/ bias
python3 ../train_HG.py HG_09_w_bias --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias vanila --epochs 10000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 ../train_HG.py HG_09_soft_lamb01 --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias vanila_ortho --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 ../train_HG.py HG_09_hard_gram --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias gram --use_gram --epochs 10000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 ../train_HG.py HG_09_hard_special --seed 0 --use_squeue --gpu 0 --data_file entropy_HG --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 16 16 16 --trunk_hidden 1 16 16 16 --use_bias depend --epochs 10000 --lambda 0



### (Toy)
### 1) Toy 1D istoropic
## vanila DeepONet wo/ bias
python3 train.py 3_8_3_8_wo_bias --seed 0 --gpu 0 --data_file toy  --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias no --epochs 100000 --lambda 0

## vanila DeepONet w/ bias
python3 train.py 3_8_3_8_w_bias --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias vanila --epochs 100000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 train.py 3_8_3_8_soft_lamb01 --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias vanila_ortho --epochs 100000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 train.py 3_8_3_8_hard_gram --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias gram --use_gram --epochs 100000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 train.py 3_8_3_8_hard_special --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias depend --epochs 100000 --lambda 0


### 2) Toy 1D HG
## vanila DeepONet wo/ bias
python3 train_HG.py 3_8_3_8_HG_wo_bias --seed 0 --gpu 0 --data_file toy  --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias no --epochs 100000 --lambda 0

## vanila DeepONet w/ bias
python3 train_HG.py 3_8_3_8_HG_w_bias --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias vanila --epochs 100000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 train_HG.py 3_8_3_8_HG_soft_lamb01 --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias vanila_ortho --epochs 100000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 train_HG.py 3_8_3_8_HG_hard_gram --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias gram --use_gram --epochs 100000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 train_HG.py 3_8_3_8_HG_hard_special --seed 0 --gpu 0 --data_file toy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 100 8 8 8 --trunk_hidden 1 8 8 8 --use_bias depend --epochs 100000 --lambda 0














#### TB

## vanila DeepONet wo/ bias
python3 trainer_tb.py wo_bias --seed 0 --gpu 0 --data_file entropy_HG_0.5 --dimension 1 --integration_order 20 --model deeponet --branch_hidden 20 100 100 8 --trunk_hidden 1 100 100 8 --use_bias no --epochs 10000 --lambda 0

## vanila DeepONet w/ bias
python3 trainer_tb.py w_bias --seed 0 --gpu 0 --data_file entropy_HG_0.5 --dimension 1 --integration_order 20 --model deeponet --branch_hidden 20 100 100 8 --trunk_hidden 1 100 100 8 --use_bias vanila --epochs 10000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 trainer_tb.py soft_lamb --seed 0 --gpu 0 --data_file entropy_HG_0.5 --dimension 1 --integration_order 20 --model deeponet --branch_hidden 20 100 100 8 --trunk_hidden 1 100 100 8 --use_bias vanila --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 trainer_tb.py hard_gram --seed 0 --gpu 0 --data_file entropy_HG_0.5 --dimension 1 --integration_order 20 --model deeponet --branch_hidden 20 100 100 8 --trunk_hidden 1 100 100 8 --use_bias no --use_gram --epochs 10000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 trainer_tb.py hard_special --seed 0 --gpu 0 --data_file entropy_HG_0.5 --dimension 1 --integration_order 20 --model deeponet --branch_hidden 20 100 100 8 --trunk_hidden 1 100 100 8 --use_bias depend --epochs 10000 --lambda 0

## steffen
python3 trainer_tb.py hard_special --seed 0 --gpu 0 --data_file entropy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 32 100 100 8 --trunk_hidden 3 100 100 8 --use_bias depend --epochs 10000 --lambda 0

