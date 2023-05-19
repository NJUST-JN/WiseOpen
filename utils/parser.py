import argparse

__all__ = ['set_parser']

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch WiseOpen Training')
    ## Computational Configurations
    parser.add_argument("--no_ood", action="store_true",
                        help="dont use ood data") 
    parser.add_argument("--ood_test", action="store_true",
                        help="whether test far ood")
   
    parser.add_argument("--S_hat_id",default=0.5,type=float,
                        help="initial threshold for pseudo ID")
    parser.add_argument("--S_max_id",default=0.9,type=float,
                        help="max threshold for pseudo ID")
    parser.add_argument("--S_min_id",default=0.5,type=float,
                        help="min threshold for pseudo ID")
    parser.add_argument("--C_id",default=1.0001,type=float,
                        help="C for pseudo ID")  
    parser.add_argument("--gama_id",default=1.05,type=float,
                        help="gama for pseudo ID")
    parser.add_argument("--th_id",default=0.5,type=float,
                        help="threshold for pseudo ID")
    
    parser.add_argument("--S_hat_ood",default=0.5,type=float,
                        help="initial threshold for pseudo OOD")
    parser.add_argument("--S_max_ood",default=0.9,type=float,
                        help="max threshold for pseudo OOD")
    parser.add_argument("--S_min_ood",default=0.5,type=float,
                        help="min threshold for pseudo OOD")
    parser.add_argument("--C_ood",default=1.0001,type=float,
                        help="C for pseudo OOD")  
    parser.add_argument("--gama_ood",default=1.05,type=float,
                        help="gama for pseudo OOD")
    parser.add_argument("--th_ood",default=0.5,type=float,
                        help="threshold for pseudo OOD")
    
    parser.add_argument('--threshold_hat', default=0.1, type=float,
                    help='for pseudo label threshold')
    parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
    parser.add_argument("--max_threshold",default=0.95,type=float,
                        help="max threshold for pseudo label")
    parser.add_argument("--min_threshold",default=0.0,type=float,
                        help="min threshold for pseudo label")
    parser.add_argument("--C_threshold",default=1.0001,type=float,
                        help="C for pseudo OOD")  
    parser.add_argument("--gama_threshold",default=1.27,type=float,
                        help="gama for pseudo OOD")
    
    parser.add_argument("--uc1",default=9,type=float,
                        help="updata cycle for id and ood th")
    parser.add_argument("--uc2",default=9,type=float,
                        help="updata cycle for threshold")
    
    parser.add_argument('--sup_warm', default=5, type=int,
                        help='epoch to warm model by supervised loss')
    parser.add_argument('--ova_warm', default=512, type=int,
                        help='epoch to use cosine annealing in alpha')
    parser.add_argument("--alpha_to",default=0.5,type=float,
                        help="used in ova focal loss")
    parser.add_argument("--alpha_from",default=0.1,type=float,
                        help="used in ova focal loss")
    parser.add_argument("--alpha",default=0.5,type=float,
                        help="used in ova focal loss")

    parser.add_argument('--least_set', default=4, type=int,
                        help='the smallest unlabled set to tarin least_set*mu*batch')
      
    parser.add_argument('--gpu_id', default='2', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no_progress', action='store_true',
                    help="don't use progress bar")
    parser.add_argument('--eval_only', type=int, default=0,
                        help='1 if evaluation mode ')
    parser.add_argument('--vis_only', action="store_true",
                        help='1 if visualization mode ')
    parser.add_argument("--kmeans", action="store_true",
                        help="use to calculate nmi")
    
    parser.add_argument('--num_classes', type=int, default=6,
                        help='for cifar10')

    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./data', type=str,
                        help='path to data directory')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny'],
                        help='dataset name')
    ## Hyper-parameters
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--num_labeled', type=int, default=400,
                        choices=[25, 50, 100, 400],
                        help='number of labeled data per each class')
    parser.add_argument('--num_val', type=int, default=50,
                        help='number of validation data per each class')
    parser.add_argument('--num_super', type=int, default=11,
                        help='number of super-class known classes cifar100: 9(5/5) or 11(6/4) or 13(7/3) or 15(8/2) or 17(9/1)')
    parser.add_argument("--expand_labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet_imagenet'],
                        help='dataset name')
    ## HP unique to OpenMatch (Some are changed from FixMatch)
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                    help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                    help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, '
                         '1.0 for CIFAR100')
    parser.add_argument('--start_fix', default=10, type=int,
                        help='epoch to start fixmatch training')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--total_steps', default=2 ** 19, type=int,
                        help='number of total steps to run')
    parser.add_argument('--epochs', default=512, type=int,
                        help='number of epochs to run')

    ##
    parser.add_argument('--eval_step', default=1024, type=int,
                        help='number of eval steps to run')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')


    args = parser.parse_args()
    return args