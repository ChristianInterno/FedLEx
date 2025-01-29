import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #set the ID devices that you want to use
import sys
import torch
import argparse
import traceback
import time


from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model

import gc
import copy

def main(args, writer):
    """Main program to run federated learning.

    Args:
        args: user input arguments parsed by argparser
        writer: `torch.utils.tensorboard.SummaryWriter` instance for TensorBoard tracking
    """
    # set seed for reproducibility
    set_seed(args.seed)

    server_dataset, client_datasets = load_dataset(args)

    # -------------------------------------------
    # Adjust device selection with graceful fallback
    # -------------------------------------------
    if 'cuda' in args.device:
        # If 'cuda' in user-supplied device, check if GPU is actually available
        if not torch.cuda.is_available():
            print("[Warning] CUDA is not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            # If device_ids are given, use the first device_id (e.g., "cuda:0", "cuda:1", etc.)
            if args.device_ids:
                args.device = f'cuda:{args.device_ids[0]}'
            else:
                args.device = 'cuda'
    elif 'mps' in args.device:
        # If 'mps' is requested, check if MPS is actually available (macOS Metal Performance Shaders)
        if not torch.backends.mps.is_available():
            print("[Warning] MPS device is not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            args.device = 'mps'
    else:
        # Otherwise, if user specifies 'cpu' or something else, default to CPU.
        args.device = 'cpu'

    # Now set up the device in PyTorch
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # get model
    model, args = load_model(args)

    # check all args before FL
    args = check_args(args)

    # create central server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model=model)

    # federated learning loop
    for curr_round in range(1, args.R + 1):
        server.round = curr_round
        selected_ids = server.update()

        if curr_round % args.eval_every == 0 or (curr_round == args.R):
            server.evaluate(excluded_ids=selected_ids)

    # wrap-up
    server.finalize()



if __name__ == "__main__":

    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    #####################
    # Default arguments #
    #####################
    parser.add_argument('--exp_name', help='experiment name', type=str, required=False, default=['FedLex'])
    seed = int(time.time())
    parser.add_argument('--seed', help='global random seed', type=int, default=seed)
    parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default='mps')
    parser.add_argument('--device_ids', nargs='+', type=int,
                        help='GPU device ids for multi-GPU training (use all available GPUs if no number is passed)',
                        default=[])
    parser.add_argument('--data_path', help='path to read data from', type=str, default='./data')
    parser.add_argument('--log_path', help='path to store logs', type=str, default='./log')
    parser.add_argument('--result_path', help='path to save results', type=str, default='./result_paper')
    parser.add_argument('--use_tb', help='use TensorBoard to track logs (atif passed)', action='store_true')
    parser.add_argument('--tb_port', help='TensorBoard port number (valid only if `use_tb`)', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address (valid only if `use_tb`)', type=str, default='0.0.0.0')

    #####################
    # Dataset arguments #
    #####################
    parser.add_argument('--dataset', help='''name of dataset to use for an experiment 
    * NOTE: case sensitive*
    - image classification datasets in `torchvision.datasets`,
    - text classification datasets in `torchtext.datasets`,
    - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
    - among [ TinyImageNet | CINIC10 | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
        ''',
                        type=str, required=False, default='CIFAR10')
    parser.add_argument('--test_fraction', help='fraction of local hold-out dataset for evaluation', type=float,
                        choices=[Range(0., 9e-1)], default=0.2)
    parser.add_argument('--rawsmpl', help='fraction of raw data to be used (only used when one of `LEAF` datasets is used)',
                        type=float, choices=[Range(0., 1.)], default=1.0)

    # data augmentation arguments
    parser.add_argument('--resize', help='resize input images (using `torchvision.transforms.Resize`)', type=int, default=24)
    parser.add_argument('--imnorm', help='normalize channels using ImageNet pre-trained mean & std (Normalize)', action='store_true')
    parser.add_argument('--randrot', help='randomly rotate input (using RandomRotation)', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip input horizontally (RandomHorizontalFlip)', type=float,
                        choices=[Range(0., 1.)], default=0.5)
    parser.add_argument('--randvf', help='randomly flip input vertically (RandomVerticalFlip)', type=float,
                        choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help='random jitter (like brightness/contrast adjustments)', type=float, default=0.5)

    # statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''type of data split scenario
    - `iid`: statistically homogeneous setting,
    - `unbalanced`: unbalance in sample counts across clients,
    - `patho`: pathological non-IID split scenario,
    - `diri`: Dirichlet distribution-based split,
    - `pre`: pre-defined data split
    ''', type=str, choices=['iid', 'unbalanced', 'patho', 'diri', 'pre'],
                        required=False, default='patho')
    parser.add_argument('--mincls', help='min number of distinct classes per client (for `patho`)', type=int, default=2)
    parser.add_argument('--cncntrtn', help='Dirichlet concentration param (for `diri`)', type=float, default=0.2)


    ###################
    # Model arguments #
    ###################
    parser.add_argument('--model_name', help='model to use (case sensitive)', type=str,
                        choices=[
                            'TwoNN', 'TwoCNN', 'FEMNISTCNN',
                            'LeNet', 'MobileNet', 'SqueezeNet',
                            'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
                            'ResNet10', 'ResNet18', 'ResNet34',
                            'MobileNeXt', 'SqueezeNeXt', 'MobileViT',
                            'NextCharLSTM', 'NextWordLSTM',
                            'DistilBert', 'SqueezeBert', 'MobileBert',
                            'LogReg', 'GRUClassifier', 'twocnn_GTL', 'BIGCNN', 'WideResNet'
                        ],
                        required=False, default='TwoCNN')
    parser.add_argument('--hidden_size', help='hidden channels for CNN or hidden dim for LMs', type=int, default=32)
    parser.add_argument('--dropout', help='dropout rate', type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--use_model_tokenizer', help='use a model-specific tokenizer if passed', action='store_true')
    parser.add_argument('--use_pt_model', help='use a pre-trained model for fine-tuning if passed', action='store_true')
    parser.add_argument('--seq_len', help='max sequence length (for torchtext)', type=int, default=512)
    parser.add_argument('--num_layers', help='number of layers in RNN cells', type=int, default=4)
    parser.add_argument('--num_embeddings', help='size of embedding dictionary', type=int, default=1000)
    parser.add_argument('--embedding_size', help='embedding dim of language models', type=int, default=512)
    parser.add_argument('--init_type', help='weight initialization method', type=str, default='xavier',
                        choices=['normal', 'xavier', 'kaiming', 'orthogonal'])
    parser.add_argument('--init_gain', type=float, default=1.0, help='magnitude of variance used for weight init')

    ######################
    # Learning arguments #
    ######################
    parser.add_argument('--algorithm', help='federated aggregation algorithm', type=str,
                        choices=['fedavg', 'fedlex'], required=False, default='fedlex')
    parser.add_argument('--eval_type', help='evaluation type (local, global, both)', type=str,
                        choices=['local', 'global', 'both'], required=False, default='both')
    parser.add_argument('--eval_fraction', help='fraction of random clients for eval', type=float,
                        choices=[Range(1e-8, 1.)], default=1.0)
    parser.add_argument('--eval_every', help='eval frequency (in rounds)', type=int, default=10)
    parser.add_argument('--eval_metrics', help='evaluation metric(s)', type=str,
                        choices=[
                            'acc1','acc5','auroc','auprc','youdenj','f1','precision','recall',
                            'seqacc','mse','mae','mape','rmse','r2','d2'
                        ],
                        nargs='+', required=False, default=['acc1','acc5','f1','precision','recall'])
    parser.add_argument('--K', help='number of total clients', type=int, default=20)
    parser.add_argument('--R', help='total number of communication rounds', type=int, default=2)
    parser.add_argument('--C', help='fraction of clients sampled each round', type=float,
                        choices=[Range(0., 1.)], default=0.2)
    parser.add_argument('--E', help='number of local epochs', type=int, default=5)
    parser.add_argument('--B', help='local batch size', type=int, default=350)
    parser.add_argument('--beta', help='global momentum factor', type=float,
                        choices=[Range(0., 1.)], default=0.9)

    # optimization arguments
    parser.add_argument('--no_shuffle', help='do not shuffle data if passed', action='store_true')
    parser.add_argument('--optimizer', help='local optimizer, submodule of torch.optim', type=str, default='Adam')
    parser.add_argument('--max_grad_norm', help='constant for grad clipping', type=float,
                        choices=[Range(0., float('inf'))], default=0.)
    parser.add_argument('--weight_decay', help='weight decay (L2)', type=float,
                        choices=[Range(0., 1.)], default=0)
    parser.add_argument('--momentum', help='momentum factor', type=float,
                        choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--lr', help='learning rate for local updates', type=float,
                        choices=[Range(0., 100.)], default=0.0003)
    parser.add_argument('--lr_server', help='learning rate for server opt', type=float,
                        choices=[Range(0., 100.)], default=0.0003)
    parser.add_argument('--lr_decay', help='decay factor for LR per round', type=float,
                        choices=[Range(0., 1.)], default=0.99)
    parser.add_argument('--lr_decay_step', help='frequency (in rounds) for LR decay', type=int, default=1)
    parser.add_argument('--criterion', help='loss function (submodule of torch.nn)', type=str, default='CrossEntropyLoss')
    parser.add_argument('--mu', help='proximity constant for fedprox-like algs', type=float,
                        choices=[Range(0., 100)], default=0.01)

    # exploration phase arguments
    parser.add_argument('--Patience_mask', help='patience for early-stopping in scouting process', type=int, default=40)
    parser.add_argument('--epoochs_mask', help='epochs for the exploration process', type=int, default=1)
    parser.add_argument('--perc_clients_for_mask', help='% of clients for which to create a mask', type=float,
                        choices=[Range(0., 1.)], default=1)
    parser.add_argument('--guidence_normalization', help='normalization type for Global Guidance mask', type=str,
                        default='MinMax')

    # extra ablation arguments (commented out in your code)
    # parser.add_argument('--sf', help='scaling factor', type=float, choices=[Range(0., 1.)], default=1)

    # Fed Optimization (Reddi et al., 2020) arguments
    parser.add_argument('--beta1', help='server momentum factor', type=float,
                        choices=[Range(0., 1.)], default=0.9)
    parser.add_argument('--tau', help='server momentum factor', type=float,
                        choices=[Range(0., 1.)], default=0.001)
    parser.add_argument('--plot', help='Do you want to plot?', default=False)
    parser.add_argument('--CEXP2', help='N', type=int, default=1)
    parser.add_argument('--random_pruning', help='Do you want random pruning?', default=False)

    # Parse arguments
    args = parser.parse_args()
    

    # parse arguments
    args = parser.parse_args()

    # make path for saving losses & metrics & models
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.exp_name = f'{args.exp_name}_{args.seed}_{args.dataset.lower()}_{args.model_name.lower()}_{curr_time}'
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}.log', args)

    # check TensorBoard execution
    if args.use_tb: # run TensorBaord for tracking losses and metrics
        tb = TensorBoardRunner(os.path.join(args.log_path, f'{args.exp_name}'), args.tb_host, args.tb_port)
    else:
        tb = None

    # define writer
    writer = SummaryWriter(
            log_dir=os.path.join(args.log_path, f'{args.exp_name}'),
            filename_suffix=f'_{curr_time}'
    )

    # run main program
    torch.autograd.set_detect_anomaly(True)
    try:
        main(args, writer)
        if args.use_tb:
            tb.finalize()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        if args.use_tb:
            tb.interrupt()
        sys.exit(1)

    # # run main program
    # main(args, writer)
    #
    # # bye!
    # if tb is not None:
    #     tb.finalize()
    # os._exit(0)


