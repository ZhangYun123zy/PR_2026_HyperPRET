import argparse
import importlib
import os
from utils import *
import modelrecords
MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='HyperPRET'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='Houston',
                        choices=['Houston','xuzhou','botswana'])
    parser.add_argument('-f', type=int, default=144,choices=[103, 144,204,80,436,200,176,145])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-net', type=str, default='mamba')

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)

    parser.add_argument('-ratio', type=float, default=0.05)
    parser.add_argument('-patch_size', type=int, default=9)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-batch_size_test', type=int, default=64)

    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    #for fact
    parser.add_argument('-balance', type=float, default=1.0)
    parser.add_argument('-loss_iter', type=int, default=200)
    parser.add_argument('-alpha', type=float, default=2.0)
    parser.add_argument('-eta', type=float, default=0.1)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-n_sessions', type=int, default=2)

    parser.add_argument('-w_BC_cls', type=float, default=5)
    parser.add_argument('-w_BC_binary', type=float, default=100)
    parser.add_argument('-w_cls', type=float, default=1, help='weight of softmax cross-entropy classification loss on novel classes')
    parser.add_argument('-w_e', type=float, default=1, help='weight of extra loss functions (e.g. CE on seen classes)')
    parser.add_argument('-w_d', type=float, default=50, help='weight of Knowledge Distillation loss')
    
    parser.add_argument('--ft_T', type=float, default=4.0, help='Knowledge Distillation temperature')
    parser.add_argument('--batch_size_current', type=int, default=5)
    parser.add_argument('--using_exemplars', action='store_true', default=True, help='save 1 exemplars per class')
    parser.add_argument('--merge_strategy', type=str, default='attn', help='how to ensmeble two branches')
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    print(args.dataset)
    if args.dataset=='Houston':
        classes_num=15
    if args.dataset=='xuzhou':
        classes_num=9
    if args.dataset=='botswana':
        classes_num=14
    pprint(vars(args))
    seeds=[1337, 1220, 1336, 1330, 1224, 1236, 1226, 1235, 1233, 1229]
    nDataSets = 1
    final_results= np.zeros([nDataSets, classes_num+3])
    final_results_my= np.zeros([nDataSets, classes_num+3])
    for  nDataSet in range(nDataSets):
        np.random.seed(seeds[nDataSet])
        torch.manual_seed(seeds[nDataSet])
        random.seed(seeds[nDataSet])
        import random
        torch.cuda.manual_seed(seeds[nDataSet])
        torch.cuda.manual_seed_all(seeds[nDataSet])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.num_gpu = set_gpu(args)
        trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
        final_results[nDataSet,:],final_results_my[nDataSet,:]=trainer.train(seeds[nDataSet])

    ELEMENT_ACC_RES_SS4=np.transpose(final_results_my[:,:classes_num])
    AA_RES_SS4=np.transpose(final_results_my[:,classes_num])
    OA_RES_SS4=np.transpose(final_results_my[:,classes_num+1])
    KAPPA_RES_SS4=np.transpose(final_results_my[:,classes_num+2])
    os.makedirs('records', exist_ok=True)
    modelrecords.outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                              classes_num, nDataSets,
                              './records/{}__{}_{}_{}_w_BC_binary_{}_w_kd_{}_new.txt'.format(args.dataset,args.net,args.balance,args.loss_iter,
                                                                                    args.w_BC_binary,args.w_d))
    
