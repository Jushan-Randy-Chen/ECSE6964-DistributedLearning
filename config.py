import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # distribute
    parser.add_argument('-P', '--mode', help='backend, default="D-SGD"',
                        dest='mode', type=str, default="D-SGD")
    parser.add_argument('-S', '--size', help='size, default=10',
                        dest='size', type=int, default=2)
 
    parser.add_argument('--rho', help='ADMM penalty parameter, default=1.0',
                        dest='rho', type=float, default=1.0)
    parser.add_argument('--consensus_tolerance', help='Consensus error tolerance for ADMM, default=1e-4',
                        dest='consensus_tolerance', type=float, default=1e-4)
    parser.add_argument('--topology', help='Network topology, default="ring"',
                        dest='topology', type=str, default="all",
                        choices=["ring", "all","meshgrid", "exponential"])
    # model
    parser.add_argument('-mn', '--model_name', help='model name, default=\'Linear\'',
                        dest='model_name', type=str, default='ResNet18')
    # dataset
    parser.add_argument('-p', '--path', help='path of dataset, default=\'../data\'',
                        dest='path', type=str, default='../data')
    parser.add_argument('-dn', '--dataset_name', help='dataset_name, default=\'MNIST\'',
                        dest='dataset_name', type=str, default='CIFAR10')
    parser.add_argument('-bs', '--batch_size', help='batch_size, default=32',
                        dest='batch_size', type=int, default=32)
    parser.add_argument('-ns', '--n_swap', help='n_swap, default=0',
                        dest='n_swap', type=int, default=None)
    # train
    parser.add_argument('-lr', '--learning_rate', help='learning_rate, default=0.01',
                        dest='lr', type=float, default=0.01)
    parser.add_argument('-mm', '--momentum', help='momentum, default=0.9',
                        dest='momentum', type=float, default=0)
    parser.add_argument('-wd', '--weight_decay', help='weight_decay, default=1e-4',
                        dest='weight_decay', type=float, default=0)
    parser.add_argument('-ne', '--num_epoch', help='num_epoch, default=100',
                        dest='num_epoch', type=int, default=100)
    parser.add_argument('-es', '--early_stop', help='early_stop, default=3000',
                        dest='early_stop', type=int, default=3000)
    parser.add_argument('-ms', '--milestones', help='scheduler, default=[30, 60, 80]', nargs='+',
                        dest='milestones', type=int, default=[])
    parser.add_argument('-g', '--gamma', help='gamma, default=0.1',
                        dest='gamma', type=float, default=0.1)
    parser.add_argument('--seed', help='seed, default=777',
                        dest='seed', type=int, default=777)
    # device
    parser.add_argument('-G', '--gpu', help='use gpu for train default True',
                        dest='gpu', type=bool, default=True)
    
    # Stability evaluation parameters
    parser.add_argument('--evaluate_stability', help='Evaluate distributed on-average stability',
                        dest='evaluate_stability', action='store_true')
    parser.add_argument('--stability_worker_subset', help='Subset of workers for stability evaluation',
                        dest='stability_worker_subset', type=int, nargs='+', default=None)
    parser.add_argument('--stability_num_samples', help='Number of samples to perturb for stability evaluation',
                        dest='stability_num_samples', type=int, default=1)

    args = parser.parse_args()
    
    # If stability_worker_subset is not provided, use the first min(5, size) workers
    if args.evaluate_stability and args.stability_worker_subset is None:
        # args.stability_worker_subset = list(range(min(5, args.size)))
        pass
        
    return args.__dict__