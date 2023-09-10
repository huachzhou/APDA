import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="APDA")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=256,
                        help="the embedding size of APDA")
    parser.add_argument('--layer', type=int,default=4,
                        help="the layer num of APDA")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='Epinions',
                        help="available datasets: [Epinions, gowalla, ifashion]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[5,10,20,50,100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--model_type', type=str, default='APDA')
    parser.add_argument('--residual_coff', type=float, default=0.6)
    parser.add_argument('--exp_coff', type=float,default=1.1)
    parser.add_argument('--exp_on', type=bool,default=False)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='APDA', help='rec-model, support [APDA]')


    return parser.parse_args()
