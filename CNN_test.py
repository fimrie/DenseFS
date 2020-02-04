# ### Imports
import molgrid

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import argparse

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Which model to use. Supported: Imrie, Ragoza. Default Imrie",default='Imrie')
    parser.add_argument('--test_file',type=str,required=True,help="Test file (types file)")
    parser.add_argument('--rotate',action='store_true',default=False,help="Add random rotations to input data")
    parser.add_argument('--translate',type=float,help="Add random translation to input data. Default 0",default=0.0)
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-b','--batch_size',type=int,required=False,help="Number of training example per iteration. Default 16",default=16)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    
    parser.add_argument('--saved_model',type=str,required=True,help="Set of weights to initialize the model with")
    parser.add_argument('--output_path',type=str,required=False,help="Path to save output",default='./output.txt')
    
    parser.add_argument('--display_iter',type=int,default=50,help='Print out network outputs every so many iterations')
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

# Use the libmolgrid ExampleProvider to load batches from a file
e = molgrid.ExampleProvider(data_root=args.data_root,balanced=False,shuffle=False)
e.populate(args.test_file)

# Load test file examples (NOTE: this should ideally be done from molgrid but not possible)
with open(args.test_file, 'r') as f:
    lines = f.readlines()

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
    model = Basic_CNN(dims).to('cuda')
elif args.model == 'Imrie':
    # Initialize Imrie Net on GPU
    model = DenseNet(dims, block_config=(4,4,4)).to('cuda')
else:
    print("Please specify a valid architecture")

print("Setup model")
print("Number of model params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

# ## Load weights for network
model.load_state_dict(torch.load(args.saved_model))
print("Loaded model parameters")

# ### Test network

# Ensure model in eval mode
model.eval()

# Test loop
predictions = []
num_samples = e.size()
num_batches = -(-num_samples // args.batch_size)
for it in range(num_batches):   
    # Load data
    batch = e.next_batch(args.batch_size)
    gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=args.translate)
    batch.extract_label(0, float_labels)
    labels = float_labels.long().to('cuda')

    # Predict
    output = F.softmax(model(input_tensor), dim=0)
    predictions.extend(list(output.detach().cpu().numpy()[:,1]))

    # Progress
    if it % args.display_iter == 0:
        print("Processed: %d / %d examples" % (it*args.batch_size, num_samples))

# Save predictions
output_lines = []
for line, pred in zip(lines, predictions):
    output_lines.append(str(pred) + ' ' + line)

with open(args.output_path, 'w') as f:
    for line in output_lines:
        f.write(line)
