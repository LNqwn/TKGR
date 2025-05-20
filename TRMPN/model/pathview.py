""" 
Path Interpretations of Predictions
Code partially adapted from authors' implementation of NBFNet
"""

import sys
import numpy as np
import torch
sys.path.append("..")
from TRMPN.utils import utils
from config import args
from TRMPN import TRMPN
from datetime import datetime, timedelta
#

def icews18ts2datetime():
    # e.g. datamap[0] = '01/01/2018'
    # e.g. datamap[303] = '10/31/2018'
    start_date = datetime(2018, 1, 1)
    d = 24
    n = 304

    # generate a dictionary of dates
    datemap = {}
    for i in range(n):
        # setting the current date
        current_date = start_date + timedelta(days=i)
        # adding the current date to the dictionary
        datemap[i] = current_date.strftime('%m/%d/%Y')
    return datemap  
#用于处理单个三元组（triplet）的路径解释
def path_process(model, history_graph, triplet, entity_vocab, relation_vocab, ts_vocab, filtered_data=None):
    # triplet: [h,r,t]
    num_relation = len(relation_vocab)
    triplet = triplet.unsqueeze(0)
    inverse = triplet[:, [2, 1, 0]]
    inverse[:, 1] += num_relation

    t_batch, h_batch = utils.all_negative(num_nodes, triplet)
    t_pred = model(history_graph, t_batch)
    h_pred = model(history_graph, h_batch)
    
    # filtered_data is the future graph
    timef_t_mask, timef_h_mask = utils.strict_negative_mask(filtered_data, triplet[: , [0, 2, 1]])
    pos_h_index, pos_r_index, pos_t_index = triplet.unbind(-1)
    timef_t_ranking = utils.compute_ranking(t_pred, pos_t_index, timef_t_mask).squeeze(0)
    timef_h_ranking = utils.compute_ranking(h_pred, pos_h_index, timef_h_mask).squeeze(0)

    samples = (triplet, inverse)
    rankings = (timef_t_ranking, timef_h_ranking)
    for sample, ranking in zip(samples, rankings):
        h, r, t = sample.squeeze(0).tolist()
        h_name = entity_vocab[h]
        t_name = entity_vocab[t]
        r_name = relation_vocab[r % num_relation]
        if r >= num_relation:
            r_name += "^(-1)"
        #模型预测：使用模型model对负样本进行预测，得到头实体和尾实体的预测分数。
        #计算排名：通过utils.compute_ranking函数计算目标三元组在负样本中的排名。
        #路径可视化：如果目标三元组的排名小于等于3（即在前3名），则调用model.visualize函数获取支持该预测的路径及其权重，并打印路径信息。
        print("\nTarget: (%s || %s) --->> %s\t<< Rank: %g >>" % (h_name, r_name, t_name, ranking))
        if ranking > 3: continue
        paths, weights = model.visualize(history_graph, sample)
        for path, weight in zip(paths, weights):
            triplets = []
            for h, t, r, ts in path:
                h_name = entity_vocab[h]
                t_name = entity_vocab[t]
                r_name = relation_vocab[r % num_relation]
                ts_name = ts_vocab[ts/24]
                if r >= num_relation:
                    r_name += "^(-1)"
                triplets.append("<%s, %s, %s, %s>" % (h_name, r_name, t_name, ts_name))
            print("\t weight: %g\n\t\t%s" % (weight, " ->\n\t\t".join(triplets)))
#用于对整个数据集进行路径解释
def path_view(args, model, data_list, num_nodes, num_rels, entity_vocab, relation_vocab, model_name = None, shownum_each_time=2):
    ts_vocab = icews18ts2datetime()
    checkpoint = torch.load(model_name, map_location=device)
    print("\nLoad Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(model_name, checkpoint['epoch'], checkpoint['args']))  # use best stat checkpoint

    # print("\nCurrent Model Parameters:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}")
    #
    # # 打印加载的权重的形状
    # print("\nCheckpoint State Dict Parameters:")
    # for name, param in checkpoint['state_dict'].items():
    #     print(f"{name}: {param.shape}")

    # 加载权重（使用 strict=False）
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model = model.to(device)
    print("\nstart enumerating")
    
    idx = [_ for _ in range(len(data_list))] # init timestamps index [0,1,2,3,...,n]
    last_future_ts = data_list[-1][0,3]

    model.eval()
    for future_sample_id in idx:
        if future_sample_id < args.history_len: continue
        # future_sample as the future graph index, i,e. [ [s,r,o],[s,r,o], ... ]
        future_list = np.array(data_list[future_sample_id], copy=True)
        np.random.shuffle(future_list)  # shuffle the future graph
        #测试集中有很多个不同时间戳下的三元组，每个时间戳下选择随机选择 shownum_each_time 20个三元组进行预测，即future_list_select
        future_list_select = future_list[:shownum_each_time,:3]
        future_ts = data_list[future_sample_id][0,3]
        # get history graph list 遍历每个未来时间戳 future_sample_id
        # 如果时间戳小于历史长度 args.history_len，则跳过（因为没有足够的历史数据）。
        #  从data_list中提取未来图数据，并随机打乱顺序。

        history_list = data_list[future_sample_id - args.history_len : future_sample_id]
        # history_list combine
        history_list = np.concatenate(history_list)

        history_graph = utils.build_history_graph(num_nodes, num_rels, history_list, device)
        future_triple = torch.from_numpy(future_list_select).long().to(device)

        time_filter_data = {
                'num_nodes': num_nodes,
                'edge_index': torch.stack([future_triple[:,0], future_triple[:,2]]),
                'edge_type': future_triple[:,1]
        }
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \
              \nfuture ts: {} / {} \t date: {} \
              \n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" \
              .format(future_ts/24, last_future_ts/24, ts_vocab[future_ts/24]))

        for triplet in future_triple:
            path_process(model, history_graph, triplet, entity_vocab, relation_vocab, ts_vocab, filtered_data=time_filter_data)
            



#调用时指定参数--pretrain_name modelYAGO.pth 直接放在model/YAGO路径下
if __name__ == '__main__':

    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args)

    model_name = "bsize:{}-neg:{}-hislen:{}-msg:{}-aggr:{}-dim:{}+{}|{}|{}|{}|{}"\
        .format(args.batch_size, args.negative_num, args.history_len, args.message_func, args.aggregate_func, 
                args.input_dim, args.hidden_dims, args.short_cut, args.layer_norm,
                args.time_encoding, args.time_encoding_independent)
    print("model-name",model_name)
    model_state_file = model_name

    # load datasets
    data = utils.load_data(args.dataset)

    if utils.get_rank() == 0:
        print("# Sanity Check: stat name : {}".format(model_state_file))
        print("# Sanity Check:  entities: {}".format(data.num_nodes))
        print("# Sanity Check:  relations: {}".format(data.num_rels))


    # change the view of the data
    # [[s,r,o,t],[s,r,o,t],[s,r,o,t],...] -->> [ [ [s,r,o,t],[s,r,o,t] ], [ [s,r,o,t] ],...]
    
    # train_list_sp = utils.split_by_time(data.train, stat_show=False)
    # valid_list_sp = utils.split_by_time(data.valid, stat_show=False)
    test_list_sp = utils.split_by_time(data.test, stat_show=False)

    num_nodes = data.num_nodes
    num_rels = data.num_rels # not include reverse edge type
    entity_vocab = data.entity_dict
    relation_vocab = data.relation_dict


    # model create
    model = TRMPN(
        args.input_dim, 
        args.hidden_dims,
        num_nodes,
        num_rels,
        message_func=args.message_func, 
        aggregate_func=args.aggregate_func,
        short_cut=args.short_cut, 
        layer_norm=args.layer_norm,
        activation="relu", 
        history_len=args.history_len,
        time_encoding=args.time_encoding,
        time_encoding_independent=args.time_encoding_independent
    )
    device = utils.get_device(args)

    model = model.to(device)
#对测试集数据调用path_view函数进行路径解释
    path_view(args, model, test_list_sp, num_nodes, num_rels, entity_vocab, relation_vocab, model_name = args.pretrain_name, shownum_each_time=20)

    sys.exit()



