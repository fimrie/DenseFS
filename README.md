# DenseFS

## Example usage

### Train from scratch (using random weights)
'''
python CNN_train.py -m Imrie --train_file ./data/small.types -d ./data/structs/ -i 501 -b 32 -s 42 --display_iter 50 --save_iter 500 --anneal_iter 100
'''

### Train from pretrained model (using existing weights)
'''
python CNN_train.py -m Imrie --train_file ./data/small.types -d ./data/structs/ -i 251 -b 32 -s 42 --display_iter 50 --save_iter 250 --anneal_iter 100 --weights model.iter-500
'''

### Test
'''
python CNN_test.py -m Imrie --saved_model model.iter-250 --test_file ./data/small.types -d ./data/structs/ -b 32 -s 42 --display_iter 50
'''
