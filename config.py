import argparse

parser = argparse.ArgumentParser(description="x")
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--save_root', type=str, default="")
parser.add_argument('--tag', type=str, default="")
parser.add_argument('--snap', type=str, default="")
parser.add_argument('--dataset',
                    type=str,
                    default="",
                    choices=['SAMM', 'CASME_2'])
parser.add_argument('--data_aug', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--seed", default=111, type=int)
parser.add_argument('--finetune_list',
                    default=[],
                    type=str,
                    nargs="+",
                    help='finetune subjects')
parser.add_argument("--patience",
                    default=15,
                    type=int,
                    help='front extend patience')

# ========================= Model Configs ==========================
parser.add_argument('--hidden_units',
                    default=[2048, 256, 256],
                    type=int,
                    nargs="+",
                    help='hidden units set up')
parser.add_argument('--length', type=int, default=64)
parser.add_argument('--step', type=int, default=64)
parser.add_argument('-L',
                    type=int,
                    default=12,
                    help='the number of input difference images')
parser.add_argument('--input_size', type=int, default=112)
parser.add_argument('--data_option',
                    type=str,
                    choices=['diff', 'wt_diff', 'wt_dr'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs',
                    default=25,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument(
    '--early_stop', type=int,
    default=3)  # if validation loss didn't improve over 3 epochs, stop
parser.add_argument('-b',
                    '--batch_size',
                    default=16,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--lr_decay_factor', default=0.1, type=float)
parser.add_argument('--lr_steps',
                    default=[2, 5],
                    type=float,
                    nargs="+",
                    metavar='LRSteps',
                    help='epochs to decay learning rate by factor')
parser.add_argument('--optim', default='SGD', type=str)
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient',
                    '--gd',
                    default=20,
                    type=float,
                    metavar='W',
                    help='gradient norm clipping (default: 20)')
parser.add_argument('--focal_alpha', default=[1., 1.], type=float, nargs="+")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 50) iteration')
parser.add_argument('--eval-freq',
                    '-ef',
                    default=1,
                    type=int,
                    metavar='N',
                    help='evaluation frequency (default: 1) epochs')

# ========================= Runtime Configs ==========================
parser.add_argument('-j',
                    '--workers',
                    default=0,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--delete_last',
                    action='store_true',
                    help='delete the last recorded subject')
parser.add_argument('-t',
                    '--test',
                    dest='test',
                    action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', type=str, default=None)
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output', type=str, default='output')
parser.add_argument('--root_runs', type=str, default='runs')
parser.add_argument('--load_pretrained', type=str, default='')
parser.add_argument('--load_bn', action='store_true')
