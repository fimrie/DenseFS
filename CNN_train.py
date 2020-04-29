# Imports
import molgrid

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os

import argparse

from sklearn.metrics import roc_auc_score

from models import Basic_CNN, DenseNet, weights_init

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Which model to use. Supported: Imrie, Ragoza. Default Imrie",default='Imrie')
    parser.add_argument('--train_file',type=str,required=True,help="Training file (types file)")
    parser.add_argument('--rotate',action='store_true',default=False,help="Add random rotations to input data")
    parser.add_argument('--translate',type=float,help="Add random translation to input data. Default 0",default=0.0)
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-i','--iterations',type=int,required=False,help="Number of iterations to run. Default 10,000",default=10000)
    parser.add_argument('-b','--batch_size',type=int,required=False,help="Number of training example per iteration. Default 16",default=16)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    
    # Optimiser settings
    parser.add_argument('--base_lr',type=float,help='Initial learning rate, default 0.01',default=0.01)
    parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9)
    parser.add_argument('--weight_decay',type=float,help="Weight decay rate, default 0.0",default=0.0)
    parser.add_argument('--anneal_rate',type=float,help="Anneal rate for learning rate. Default 0.9",default=0.9)
    parser.add_argument('--anneal_iter',type=float,help="How frequently to anneal learning rate (iterations). Default 5000",default=5000)
    parser.add_argument('--clip_gradients',type=float,help="Clip gradients threshold (default 10)",default=10.0)
    
    parser.add_argument('--display_iter',type=int,help='Print out network outputs every so many iterations',default=100)

    # Load pretrained model
    parser.add_argument('--weights',type=str,help="Set of weights to initialize the model with")

    # Saving
    parser.add_argument('--save_dir',type=str,required=False,help="Directory to save models",default='./')
    parser.add_argument('--save_iter',type=int,help="How frequently to save (iterations). Default 5000",default=5000)
    parser.add_argument('--save_prefix',type=str,help="Prefix for saved model, default <prefix/model>.iter-<iterations>",default='model')

    # Testing on validation set (optional)
    parser.add_argument('--test_file',type=str,help="Test file (types file)",default='')
    parser.add_argument('--test_iter',type=int,help="How frequently to test",default=1000)
    parser.add_argument('--num_rotate',type=int,help="Number of random rotations to perform during testing",default=1)

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
    e = molgrid.ExampleProvider(data_root=args.data_root, balanced=True, shuffle=True)
    e.populate(args.train_file)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (args.batch_size,)+dims

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

    # Set weights for network
    if args.weights:
        model.load_state_dict(torch.load(args.weights))
        print("Loaded model parameters")
    else:
        model.apply(weights_init)
        print("Randomly initialised model parameters")

    # Print number of parameters in model
    print("Number of model params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000))

    # Train network

    # Construct optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print("Initial learning rate: %.6f" % scheduler.get_lr()[0])

    # Train loop 
    losses = []
    for it in range(1, args.iterations+1):   
        # Load data
        batch = e.next_batch(args.batch_size)
        gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=args.translate)
        batch.extract_label(0, float_labels)
        labels = float_labels.long().to('cuda')

        # Train
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output,labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
        optimizer.step()
        losses.append(float(loss))

        # Anneal learning rate
        if it % args.anneal_iter == 0:
            scheduler.step()
            print("Current iteration: %d, Annealing learning rate: %.6f" % (it, scheduler.get_lr()[0]))

        # Progress
        if it % args.display_iter == 0:
            print("Current iteration: %d, Loss: %.3f" % (it, float(np.mean(losses[-args.display_iter:]))))

        # Save model
        if it % args.save_iter == 0:
            print("Saving model after %d iterations." % it)
            torch.save(model.state_dict(), args.save_dir + "/" + args.save_prefix + ".iter-" + str(it))

        # Test model
        if args.test_file != '' and it % args.test_iter == 0:
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
                batch.extract_label(0, float_labels)
                labs.extend(list(float_labels.detach().cpu().numpy()))
                for _ in range(args.num_rotate):
                    gmaker.forward(batch, input_tensor, random_rotation=args.rotate, random_translation=0.0)
                    # Predict
                    output = F.softmax(model(input_tensor), dim=1)
                    batch_predictions.append(list(output.detach().cpu().numpy()[:,0]))
                predictions.extend(list(np.mean(batch_predictions, axis=0)))
            # Print performance
            labs = labs[:num_samples]
            predictions = predictions[:num_samples]
            print("Current iter: %d, AUC: %.2f" % (it, roc_auc_score(labs, predictions)), flush=True)
            # Set to train mode
            model.train()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
