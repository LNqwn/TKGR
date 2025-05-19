from TGATN.models import base_model
from TGATN.utils.utils import EnhancedDict,gpu_setting
import argparse

def main(initial_dict:dict):
    opts = EnhancedDict(initial_dict)
    path = './data/' + opts.dataset + '/'

    opts.path = path



    opts.disable_bar = False
    opts.tag = f"L{opts.n_layer}"+opts.tag
    
    # 自动选择合适的GPU
    gpu_setting(opts.get("gpu",-1))
    #
    if opts.train_mode.lower() == 'base':
        trainer = base_model.Trainer(opts)
    elif opts.train_mode.lower() == 'half':
        trainer = base_model.HalfTrainer(opts)
    else:
        raise Exception("未实现的训练模式")
    for epoch in range(opts.epochs):
        trainer.train_epoch()
        """if epoch > 0:
            if model.train_history[-1][1] < model.train_history[-2][1]:
                decline_step = decline_step + 1
            else:
                decline_step = 0
            if decline_step >= stop_step:
                print('best : mrr ',model.train_history[-stop_step][1],' hist@1 ',model.train_history[-stop_step][2],' hist@10 ',model.train_history[-stop_step][3])
                break"""
    trainer.process_results()

    

if __name__ == '__main__':
    # 1. 定义命令行解析器对象
    parser = argparse.ArgumentParser(description='Demo of argparse')
    
    # 2. 添加命令行参数
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--dataset', type=str, default="ICEWS14s")
    parser.add_argument('--batch_size', type=int, default=128, help="略")
    parser.add_argument('--single_timestamp_layer_numbers', type=int, default=2)
    parser.add_argument('--gnn_mode', type=str, default="unique" )
    parser.add_argument('--time_mode', type=str, default="embedding")
    parser.add_argument('--model_name', type=str, default="TRED_GNN")
    parser.add_argument('--window_size', type= int, default=10)
    parser.add_argument('--hidden_dim', type= int, default=64)
    parser.add_argument('--max_global_window_size', type= int, default=5000)
    parser.add_argument('--epochs', type= int, default=10)
    parser.add_argument('--gpu', type= int, default=-1)
    parser.add_argument('--tag', type= str, default='')
    parser.add_argument('--train_mode', type= str, default='base')
    parser.add_argument('--lr', type= float, default=0.005)
    parser.add_argument('--attention_dim', type= int, default=5)
    parser.add_argument('--act', type= str, default="idd", choices=['idd', 'relu', 'tanh'])
    parser.add_argument('--lamb', type= float, default=0.00012)
    parser.add_argument('--dropout', type= float, default=0.25)
    parser.add_argument('--time_dim', type=int, default=16)
    # 3. 从命令行中结构化解析参数
    args = parser.parse_args()
    main(initial_dict=vars(args))
    