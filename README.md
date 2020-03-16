# DenseFS

## Example usage

## Classification

### Train from scratch (using random weights)
```
python CNN_train.py -m Imrie --train_file ./data/small.types -d ./data/structs/ -i 501 -b 32 -s 42 --display_iter 50 --save_iter 500 --anneal_iter 100
```

### Train from pretrained model (using existing weights)
```
python CNN_train.py -m Imrie --train_file ./data/small.types -d ./data/structs/ -i 251 -b 32 -s 42 --display_iter 50 --save_iter 250 --anneal_iter 100 --weights model.iter-500
```

### Test
```
python CNN_test.py -m Imrie --saved_model model.iter-250 --test_file ./data/small.types -d ./data/structs/ -b 32 -s 42 --display_iter 50
```

## Regression

### Train
#### Regualar
```
python CNN_regression_train.py -m Imrie --train_file ./data/small.types -d ./data/structs/ -i 5001 -b 32 -s 42 --label_idx 1 --save_prefix Imrie_regression --save_iter 1000 --save_dir ./ --display_iter 500 --anneal_iter 2500 --rotate --translate 2.0 --base_lr 0.001 --weight_decay 0.001 --num_rotate 1 --test_file ./data/small.types --test_iter 1000
```
#### With random augmentation
```
python CNN_regression_random_augmentation_train.py -m Imrie --train_file ./data/small.types -d ./data/structs/ -i 5001 -b 32 -s 42 --label_idx 1 --save_prefix Imrie_regression --save_iter 1000 --save_dir ./ --display_iter 500 --anneal_iter 2500 --rotate --translate 2.0 --base_lr 0.001 --weight_decay 0.001 --num_rotate 1 --test_file ./data/small.types --test_iter 1000
```

### Test
```
python CNN_regression_test.py -m Imrie --test_file ./data/small.types -d ./data/structs/ -b 32 -s 42 --saved_model Imrie_regression.iter-5000 --output_path Imrie_regression.iter-5000-preds.txt
```
