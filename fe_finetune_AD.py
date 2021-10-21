import sys
from PT_HGNN.data import *
from PT_HGNN.model import *
from warnings import filterwarnings
filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Fine-Tuning on OAG Author Disambiguation (AD) task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='The address of preprocessed graph.')
parser.add_argument('--use_pretrain', help='Whether to use pre-trained model', action='store_true')
parser.add_argument('--pretrain_model_dir', type=str, default='gpt_all_cs',
                    help='The address for pretrained model.')
parser.add_argument('--model_dir', type=str, default='models',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--dif_model_dir', type=str, default='my',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='AD',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=2,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)')  
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--scheduler', type=str, default='cycle',
                    help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
parser.add_argument('--data_percentage', type=float, default=0.1,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=16,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient Norm Clipping')
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)') 

args = parser.parse_args()
args_print(args)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

if not os.path.isdir(os.path.join(args.dif_model_dir, args.domain)):
    os.mkdir(os.path.join(args.dif_model_dir, args.domain))

print('Start Loading Graph Data...')
graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
print('Finish Loading Graph Data!')

target_type = 'paper'

types = graph.get_types()

apd = graph.edge_list['author']['paper']['rev_AP_write_other']
author_dict = {i : True for i in apd if len(apd[i]) > 1}

name_count = defaultdict(lambda: [])
# total = len(graph.node_feature['author'])
for i, j in graph.node_feature['author'].iterrows():
    if i in author_dict:
        name_count[j['name']] += [i]
name_count = {name: name_count[name] for name in name_count if len(name_count[name]) > 2}

pre_range   = {t: True for t in graph.times if t != None and t < 2014}
train_range = {t: True for t in graph.times if t != None and t >= 2014  and t <= 2016}
valid_range = {t: True for t in graph.times if t != None and t > 2016  and t <= 2017}
test_range  = {t: True for t in graph.times if t != None and t > 2017}

train_pairs = {}
valid_pairs = {}
test_pairs  = {}
'''
    Prepare all the author with same name and their written papers.
'''

for name in name_count:
    same_name_author_list = np.array(name_count[name])
    for author_id, author in enumerate(same_name_author_list):
        for p_id in graph.edge_list['author']['paper']['rev_AP_write_other'][author]:
            _time = graph.edge_list['author']['paper']['rev_AP_write_other'][author][p_id]
            if type(_time) != int:
                continue
            if _time in train_range:
                if name not in train_pairs:
                    train_pairs[name] = []
                train_pairs[name] += [[p_id, author_id, _time]]
            elif _time in valid_range:
                if name not in valid_pairs:
                    valid_pairs[name] = []
                valid_pairs[name] += [[p_id, author_id, _time]]
            elif _time in test_range:
                if name not in test_pairs:
                    test_pairs[name]  = []
                test_pairs[name]  += [[p_id, author_id, _time]]

                
def mask_softmax(pred, size):
    loss = 0
    stx = 0
    num = 0
    for l in size:
        loss += torch.log_softmax(pred[stx: stx + l], dim=-1)[0] / np.log(l)
        stx  += l
        num  += 1
    return -loss / num




def author_disambiguation_sample(seed, pairs, time_range, batch_size):
    '''
        sub-graph sampling and label preparation for author disambiguation:
        (1) Sample batch_size // 4 number of names
    '''
    np.random.seed(seed)
    names = np.random.choice(list(pairs.keys()), batch_size // 4, replace = True)
    '''
        (2) Get all the papers written by these same-name authors, and then prepare the label
    '''

    author_dict = {}
    author_info = []
    paper_info  = []
    name_label  = []
    max_time    = np.max(list(time_range.keys()))

    for name in names:
        author_list = name_count[name]
        for a_id in author_list:
            if a_id not in author_dict:
                author_dict[a_id] = len(author_dict)
                author_info += [[a_id, max_time]]
        for p_id, author_id, _time in pairs[name]:
            paper_info  += [[p_id, _time]]
            '''
                For each paper, create a list: the first entry is the true author's id, 
                while the others are negative samples (id of authors with same name)
            '''
            name_label  +=  [[author_dict[author_list[author_id]]] + \
                [author_dict[a_id] for (x_id, a_id) in enumerate(author_list) if x_id != author_id]]


    '''
        (3) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, \
                inp = {'paper': np.array(paper_info), 'author': np.array(author_info)}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width)


    '''
        (4) Mask out the edge between the output target nodes (paper) with output source nodes (author)
    '''
    masked_edge_list = []
    for i in edge_list['paper']['author']['AP_write_other']:
        if i[0] >= batch_size:
            masked_edge_list += [i]
    edge_list['paper']['author']['AP_write_other'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['author']['paper']['rev_AP_write_other']:
        if i[1] >= batch_size:
            masked_edge_list += [i]
    edge_list['author']['paper']['rev_AP_write_other'] = masked_edge_list
    
    '''
        (5) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    '''
        (6) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = {}
    for x_id, author_ids in enumerate(name_label):
        ylabel[x_id + node_dict['paper'][0]] = np.array(author_ids) + node_dict['author'][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, ylabel


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(author_disambiguation_sample, args=(randint(), \
            sel_train_pairs, train_range, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(author_disambiguation_sample, args=(randint(), \
            sel_valid_pairs, valid_range, args.batch_size))
    jobs.append(p)
    return jobs


seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

'''
    Only train and valid with a certain percentage of data, if necessary.
'''
sel_train_pairs = {p : train_pairs[p] for p in np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace = False)}
sel_valid_pairs = {p : valid_pairs[p] for p in np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace = False)}


'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn =  GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]) + 401, n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm)
if args.use_pretrain:
    gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict = False)
    print('Load Pre-trained Model from (%s)' % args.pretrain_model_dir)
matcher = Matcher(args.n_hid, args.n_hid)

for name, para in gnn.named_parameters():
    para.requires_grad_(False)

model = nn.Sequential(gnn, matcher).to(device)

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(pg, lr = 5e-4)

stats = []
res = []
best_val   = 0
train_step = 0

pool = mp.Pool(8)
st = time.time()
jobs = prepare_data(pool)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(8)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train (time < 2015)
    '''
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        for node_feature, node_type, edge_time, edge_index, edge_type, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))

            author_key = []
            paper_key  = []
            key_size   = []
            for paper_id in ylabel:
                author_ids  = ylabel[paper_id]
                paper_key  += [np.repeat(paper_id, len(author_ids))]
                author_key += [author_ids]
                key_size   += [len(author_ids)]
            paper_key  = torch.LongTensor(np.concatenate(paper_key)).to(device)
            author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

            train_paper_vecs  = node_rep[paper_key]
            train_author_vecs = node_rep[author_key]
            res = matcher.forward(train_author_vecs, train_paper_vecs)
            loss = mask_softmax(res, key_size)


            optimizer.zero_grad() 
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
    '''
        Valid (2015 <= time <= 2016)
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))

        author_key = []
        paper_key  = []
        key_size   = []
        for paper_id in ylabel:
            author_ids  = ylabel[paper_id]
            paper_key  += [np.repeat(paper_id, len(author_ids))]
            author_key += [author_ids]
            key_size   += [len(author_ids)]
        paper_key  = torch.LongTensor(np.concatenate(paper_key)).to(device)
        author_key = torch.LongTensor(np.concatenate(author_key)).to(device)
        
        valid_paper_vecs  = node_rep[paper_key]
        valid_author_vecs = node_rep[author_key]
        res = matcher.forward(valid_author_vecs, valid_paper_vecs)
        loss = mask_softmax(res, key_size)
        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        ser = 0
        for s in key_size:
            p = res[ser: ser + s]
            l = torch.zeros(s)
            l[0] = 1
            r = l[p.argsort(descending = True)]
            valid_res += [r.cpu().detach().tolist()]
            ser += s
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
        valid_mrr  = np.average(mean_reciprocal_rank(valid_res))
        
        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.dif_model_dir, args.domain, args.pretrain_model_dir.split('/')[-1] + args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f  Valid MRR: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                    loss.cpu().detach().tolist(), valid_ndcg, valid_mrr))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data


# '''
#     Evaluate the trained model via test set (time > 2016)
# '''

best_model = torch.load(os.path.join(args.dif_model_dir, args.domain, args.pretrain_model_dir.split('/')[-1] + args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, matcher = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, ylabel = \
                    author_disambiguation_sample(randint(), test_pairs, test_range, args.batch_size)
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        
        author_key = []
        paper_key  = []
        key_size   = []
        for paper_id in ylabel:
            author_ids  = ylabel[paper_id]
            paper_key  += [np.repeat(paper_id, len(author_ids))]
            author_key += [author_ids]
            key_size   += [len(author_ids)]
        paper_key  = torch.LongTensor(np.concatenate(paper_key)).to(device)
        author_key = torch.LongTensor(np.concatenate(author_key)).to(device)
        
        test_paper_vecs  = node_rep[paper_key]
        test_author_vecs = node_rep[author_key]
        res = matcher.forward(test_author_vecs, test_paper_vecs)

        ser = 0
        for s in key_size:
            p = res[ser: ser + s]
            l = torch.zeros(s)
            l[0] = 1
            r = l[p.argsort(descending = True)]
            test_res += [r.cpu().detach().tolist()]
            ser += s
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    print('Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    print('Test MRR:  %.4f' % np.average(test_mrr))
