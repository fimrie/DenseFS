# Imports
import molgrid

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import argparse

from sklearn.metrics import roc_auc_score

from models import Basic_CNN, DenseNet, weights_init

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Which model to use. Supported: Imrie, Ragoza. Default Imrie",default='Imrie')
    parser.add_argument('--test_file',type=str,required=True,help="Test file (types file)")
    parser.add_argument('--rotate',action='store_true',default=False,help="Add random rotations to input data")
    parser.add_argument('--translate',type=float,help="Add random translation to input data. Default 0",default=0.0)
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-b','--batch_size',type=int,required=False,help="Number of training example per iteration. Default 16",default=16)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    
    parser.add_argument('--weights',type=str,required=True,help="Set of weights to initialize the model with")
    parser.add_argument('--output_path',type=str,required=False,help="Path to save output",default='./output.txt')
    
    parser.add_argument('--display_iter',type=int,default=50,help='Print out network outputs every so many iterations')

    parser.add_argument('--num_rotate',type=int,help="Number of random rotations to perform during testing",default=1)
    
    parser.add_argument('--evaluate',action='store_true',default=False,help="Evaluate performance using AUCROC")
    args = parser.parse_args(argv)
    
    return args

def main(args):
    # Fix seeds
    molgrid.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set CuDNN options for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up libmolgrid
    e = molgrid.ExampleProvider(data_root=args.data_root,balanced=False,shuffle=False)
    e.populate(args.test_file)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (args.batch_size,)+dims

    # Load test file examples (NOTE: not possible to do directly via molgrid)
    with open(args.test_file, 'r') as f:
        lines = f.readlines()

    # Construct input tensors
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
    float_labels = torch.zeros(args.batch_size, dtype=torch.float32)

    # Initialise network - Two models currently available (see models.py for details)
    if args.model == 'Ragoza':
        model = Basic_CNN(dims).to('cuda')
    elif args.model == 'Imrie':
        model = DenseNet(dims, block_config=(4,4,4)).to('cuda')
    else:
        print("Please specify a valid architecture")
        exit()
    # Load weights for network
    model.load_state_dict(torch.load(args.weights))
    print("Loaded model parameters")

    # Print number of parameters in model
    print("Number of model params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    # Test network

    # Ensure model in eval mode
    model.eval()

    # Test loop
    predictions = []
    labels = []
    num_samples = e.size()
    num_batches = -(-num_samples // args.batch_size)
    print("Number of examples: %d" % num_samples)
    for it in range(num_batches):   
        # Load data
        batch = e.next_batch(args.batch_size)
        gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=args.translate)
        batch.extract_label(0, float_labels)
        labels.extend(list(float_labels.detach().cpu().numpy()))
        batch_predictions = []
        for _ in range(args.num_rotate):
            gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=args.translate)
            # Predict
            output = F.softmax(model(input_tensor), dim=1)
            batch_predictions.append(list(output.detach().cpu().numpy()[:,1]))
        predictions.extend(list(np.mean(batch_predictions, axis=0)))

        # Progress
        if it % args.display_iter == 0:
            print("Processed: %d / %d examples" % (it*args.batch_size, num_samples))

    # Print performance
    labels = labels[:num_samples]
    predictions = predictions[:num_samples]
    if args.evaluate:
        print("Test AUC: %.2f" % (roc_auc_score(labels, predictions)), flush=True)

    # Save predictions
    output_lines = []
    for line, pred in zip(lines, predictions):
        output_lines.append(str(pred) + ' ' + line)

    with open(args.output_path, 'w') as f:
        for line in output_lines:
            f.write(line)

if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
