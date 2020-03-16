# ### Imports
import molgrid

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os

import argparse

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Which model to use. Supported: Imrie, Ragoza. Default Imrie",default='Imrie')
    parser.add_argument('--train_file',type=str,required=True,help="Training file (types file)")
    parser.add_argument('--rotate',action='store_true',default=False,help="Add random rotations to input data")
    parser.add_argument('--translate',type=float,help="Add random translation to input data. Default 0",default=0.0)
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-i','--iterations',type=int,required=False,help="Number of iterations to run. Default 10,000",default=10000)
    parser.add_argument('-b','--batch_size',type=int,required=False,help="Number of training example per iteration. Default 16",default=16)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    
    parser.add_argument('--label_idx',type=int,help="Idx of label to use, default 0",default=0)
    
    parser.add_argument('--base_lr',type=float,help='Initial learning rate, default 0.01',default=0.01)
    parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9)
    parser.add_argument('--weight_decay',type=float,help="Weight decay rate, default 0.0",default=0.0)
    parser.add_argument('--anneal_rate',type=float,help="Anneal rate for learning rate. Default 0.9",default=0.9)
    parser.add_argument('--anneal_iter',type=float,help="How frequently to anneal learning rate (iterations). Default 5000",default=5000)
    
    parser.add_argument('--weights',type=str,help="Set of weights to initialize the model with")
    
    parser.add_argument('--clip_gradients',type=float,default=10.0,help="Clip gradients threshold (default 10)")
    parser.add_argument('--display_iter',type=int,default=50,help='Print out network outputs every so many iterations')

    parser.add_argument('--test_file',type=str,help="Test file (types file)", default='')
    parser.add_argument('--test_iter',type=int,help="How frequently to test", default=1000)
    parser.add_argument('--num_rotate',type=int,default=1,help="Number of random rotations to perform")

    # Saving
    parser.add_argument('--save_dir',type=str,required=False,help="Directory to save models",default='./')
    parser.add_argument('--save_iter',type=int,help="How frequently to save (iterations). Default 5000",default=5000)
    parser.add_argument('--save_prefix',type=str,help="Prefix for saved model, default <prefix/model>.iter-<iterations>",default='')

    args = parser.parse_args(argv)
    
    return args

# ### Setup
args = parse_args()
print(args)

# Fix seeds
molgrid.set_random_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Setting CuDNN options for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ### Helper initialisation function

# Define weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


# ### Setup libmolgrid

# Use the libmolgrid ExampleProvider to obtain shuffled, balanced, and stratified batches from a file
e = molgrid.ExampleProvider(data_root=args.data_root,balanced=False,shuffle=True)
e.populate(args.train_file)

# Test provider
if args.test_file != '':
    e_test = molgrid.ExampleProvider(data_root=args.data_root,balanced=False,shuffle=False)
    e_test.populate(args.test_file)

# Initialize libmolgrid GridMaker
gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.num_types())
tensor_shape = (args.batch_size,)+dims

# Construct input tensors
input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(args.batch_size, dtype=torch.float32)

# ### Setup network

from models import Basic_CNN, DenseNet

if args.model == 'Ragoza':
    # Initialize Ragoza Net on GPU
    model = Basic_CNN(dims, num_classes=1).to('cuda')
elif args.model == 'Imrie':
    # Initialize Imrie Net on GPU
    model = DenseNet(dims, block_config=(4,4,4), num_classes=1).to('cuda')
else:
    print("Please specify a valid architecture")

# Set weights for network
if args.weights:
    model.load_state_dict(torch.load(args.weights))
    print("Loaded model parameters")
else:
    model.apply(weights_init)
    print("Randomly initialised model parameters")

print("Loaded model")
print("Number of model params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

# ### Train network

# Construct optimizer
optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
print("Initial learning rate: %.6f" % scheduler.get_lr()[0])

# Train loop 
losses = []
for it in range(args.iterations):   
    # Load data
    batch = e.next_batch(args.batch_size)
    gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=args.translate)

    batch.extract_label(args.label_idx, float_labels)
    labels = float_labels.to('cuda').view(args.batch_size,1)

    # Train
    optimizer.zero_grad()
    output = model(input_tensor).view(args.batch_size,1)
    loss = F.mse_loss(output,labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
    optimizer.step()
    losses.append(float(loss))

    # Anneal learning rate
    if it % args.anneal_iter == 0 and it>0:
        scheduler.step()
        print("Current iteration: %d, Annealing learning rate: %.6f" % (it, scheduler.get_lr()[0]))

    # Progress
    if it % args.display_iter == 0:
        print("Current iteration: %d, Loss: %.3f" % (it, float(np.mean(losses[-args.display_iter:]))))
    # Save model
    if it % args.save_iter == 0 and it>0:
        print("Saving model after %d iterations." % it)
        if args.save_prefix == '':
            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(it))
        else:
            torch.save(model.state_dict(), args.save_dir + "/" + args.save_prefix + ".iter-" + str(it))

    # Test model
    if args.test_file != '' and it % args.test_iter == 0 and it>0:
        # Test loop
        # Set to test mode
        model.eval()
        predictions = []
        labs = []
        e_test = molgrid.ExampleProvider(data_root=args.data_root,balanced=False,shuffle=False)
        e_test.populate(args.test_file)
        num_samples = e_test.size()
        num_batches = -(-num_samples // args.batch_size)
        for _ in range(num_batches):
            # Load data
            batch = e_test.next_batch(args.batch_size)
            batch_predictions = []
            batch.extract_label(args.label_idx, float_labels)
            labs.extend(list(float_labels.detach().cpu().numpy()))
            for _ in range(args.num_rotate):
                gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=0.0)
                # Predict
                output = model(input_tensor)
                batch_predictions.append(list(output.detach().cpu().numpy()[:,0]))
            predictions.extend(list(np.mean(batch_predictions, axis=0)))
        # Print performance
        labs = labs[:num_samples]
        predictions = predictions[:num_samples]
        print("Current iter: %d, Pearson correlation: %.2f, RMSE: %.2f" % (it, pearsonr(labs, predictions)[0], sqrt(mean_squared_error(labs, predictions))), flush=True)
        # Set to train mode
        model.train()
