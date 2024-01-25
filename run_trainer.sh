#### JaeYong

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


### 3) Toy 3D isotropic
## vanila DeepONet wo/ bias
python3 train.py 3_16_3_16_wo_bias --seed 0 --gpu 0 --data_file toy  --dimension 3 --integration_order 4 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias no --epochs 100000 --lambda 0

## vanila DeepONet w/ bias
python3 train.py 3_16_3_16_w_bias --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias vanila --epochs 100000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 train.py 3_16_3_16_soft_lamb01 --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias vanila_ortho --epochs 100000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 train.py 3_16_3_16_hard_gram --seed 0 --data_file toy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias gram --use_gram --epochs 100000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 train.py 3_16_3_16_hard_special --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 4 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias depend --epochs 100000 --lambda 0


### 4) Toy 3D HG
## vanila DeepONet wo/ bias
python3 train_HG.py 3_16_3_16_HG_wo_bias --seed 0 --gpu 0 --data_file toy  --dimension 3 --integration_order 10 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias no --epochs 100000 --lambda 0

## vanila DeepONet w/ bias
python3 train_HG.py 3_16_3_16_HG_w_bias --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 10 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias vanila --epochs 100000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 train_HG.py 3_16_3_16_HG_soft_lamb01 --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 10 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias vanila_ortho --epochs 100000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 train_HG.py 3_16_3_16_HG_hard_gram --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 10 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias gram --use_gram --epochs 100000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 train_HG.py 3_16_3_16_HG_hard_special --seed 0 --gpu 0 --data_file toy --dimension 3 --integration_order 10 --model deeponet --branch_hidden 200 16 16 16 --trunk_hidden 3 16 16 16 --use_bias depend --epochs 100000 --lambda 0


### 5) Entropy 1D isotropic
## vanila DeepONet wo/ bias
python3 train.py 3_64_3_64_wo_bias --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 32 64 64 64 --trunk_hidden 1 64 64 64 --use_bias no --epochs 10000 --lambda 0

## vanila DeepONet w/ bias
python3 train.py 3_64_3_64_w_bias --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 32 64 64 64 --trunk_hidden 1 64 64 64 --use_bias vanila --epochs 10000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 train.py 3_64_3_64_soft_lamb01 --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 32 64 64 64 --trunk_hidden 1 64 64 64 --use_bias vanila_ortho --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 train.py 3_64_3_64_hard_gram --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 32 64 64 64 --trunk_hidden 1 64 64 64 --use_bias gram --use_gram --epochs 10000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 train.py 3_64_3_64_hard_special --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --model deeponet --branch_hidden 32 64 64 64 --trunk_hidden 1 64 64 64 --use_bias depend --epochs 10000 --lambda 0


### 6) Entropy 1D HG
## vanila DeepONet wo/ bias
python3 train_HG.py 3_64_3_64_HG_wo_bias --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 64 64 64 --trunk_hidden 1 64 64 64 --use_bias no --epochs 10000 --lambda 0

## vanila DeepONet w/ bias
python3 train_HG.py 3_64_3_64_HG_w_bias --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 64 64 64 --trunk_hidden 1 64 64 64 --use_bias vanila --epochs 10000 --lambda 0

## (soft constraint) DeepONet with additional orthogonal loss
python3 train_HG.py 3_64_3_64_HG_soft_lamb01 --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 64 64 64 --trunk_hidden 1 64 64 64 --use_bias vanila_ortho --epochs 10000 --lambda 0.1

## (Hard constraint) DeepONet with gram schmidt for basis
python3 train_HG.py 3_64_3_64_HG_hard_gram --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 64 64 64 --trunk_hidden 1 64 64 64 --use_bias gram --use_gram --epochs 10000 --lambda 0

## (Hard constraint) DeepONet with special bias
python3 train_HG.py 3_64_3_64_HG_hard_special --seed 0 --gpu 0 --data_file entropy --dimension 1 --integration_order 100 --aniso_param 0.9 --model deeponet --branch_hidden 100 64 64 64 --trunk_hidden 1 64 64 64 --use_bias depend --epochs 10000 --lambda 0












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