from .conv import *
import random
import numpy as np
from gensim.parsing.preprocessing import *
import time

from collections import defaultdict
from .cython_util import *


class PT_HGNN(nn.Module):
    def __init__(self, gnn, rem_edge_list, attr_decoder, types, neg_samp_num, device, neg_queue_size = 0):
        super(PT_HGNN, self).__init__()
        if gnn is None:
            return
        self.types = types
        self.gnn = gnn
        self.params = nn.ModuleList()
        self.neg_queue_size = neg_queue_size
        self.link_dec_dict = {}
        self.neg_queue = {}
        for source_type in rem_edge_list:
            self.link_dec_dict[source_type] = {}
            self.neg_queue[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                print(source_type, relation_type)
                matcher = Matcher(gnn.n_hid, gnn.n_hid)
                self.neg_queue[source_type][relation_type] = torch.FloatTensor([]).to(device)
                self.link_dec_dict[source_type][relation_type] = matcher
                self.params.append(matcher)
        
        self.attr_decoder = attr_decoder
        self.init_emb = nn.Parameter(torch.randn(gnn.in_dim))
        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.neg_samp_num = neg_samp_num

        # additional part (structure level)
        self.f_k = nn.Bilinear(gnn.n_hid, gnn.n_hid, 1)
        torch.nn.init.xavier_uniform_(self.f_k.weight.data)
        self.sigm = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.neg_structure_queue_size = 127
        self.neg_structure_queue = torch.randn(self.neg_structure_queue_size, gnn.n_hid).to(device)
        self.device = device
        self.structure_map_dict = {}
        self.structure_matcher = Matcher(gnn.n_hid, gnn.n_hid)
        
        for source_type in rem_edge_list:
            self.structure_map_dict[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                structure_map = StructureMapping(gnn.n_hid, gnn.n_hid)
                self.structure_map_dict[source_type][relation_type] = structure_map
                self.params.append(structure_map)

        
    def neg_sample(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = negative_sample(souce_node_list, pos_node_list, self.neg_samp_num)
        return neg_nodes
    
    def neg_sample_ori(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = []
        keys = {key : True for key in pos_node_list}
        tot  = 0
        for node_id in souce_node_list:
            if node_id not in keys:
                neg_nodes += [node_id]
                tot += 1
            if tot == self.neg_samp_num:
                break     
        return neg_nodes   

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        return self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)

    def link_loss(self, node_emb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False, relation_level = False):
        losses = 0
        ress   = []
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type]
                if len(rem_edges) <= 8:
                    continue
                
                # if relation_level:
                #     ori_edges = []
                #     for ori_source_type in rem_edge_list:
                #         if ori_source_type not in self.link_dec_dict:
                #             continue                    
                #         for ori_relation_type in rem_edge_list[ori_source_type]:
                #             tmp_edge = ori_edge_list[ori_source_type][ori_relation_type].copy()
                #             tmp_edge[:, 1] = tmp_edge[:, 1] + node_dict[ori_source_type][0]
                #             ori_edges.append(tmp_edge)
                #             del tmp_edge
                #     ori_edges = np.vstack(ori_edges)
                # else:
                #     ori_edges = ori_edge_list[source_type][relation_type]
                if relation_level:
                    ori_edges = ori_edge_list[source_type][relation_type]
                    ori_edges[:, 1] += node_dict[source_type][0]
                else:
                    ori_edges = ori_edge_list[source_type][relation_type]
                # ori_edges = ori_edge_list[source_type][relation_type]
                matcher = self.link_dec_dict[source_type][relation_type]

                # with torch.no_grad():
                target_ids, positive_source_ids = rem_edges[:,0].reshape(-1, 1), rem_edges[:,1].reshape(-1, 1)
                n_nodes = len(target_ids)
                # source_node_ids contains the all negative nodes, which we select the negative from it
                source_node_ids = np.unique(ori_edges[:, 1])

                # 直接在数据里面进行采样的操作，对应的就是采样的结果
                negative_source_ids = [self.neg_sample(source_node_ids, \
                    ori_edges[ori_edges[:, 0] == t_id][:, 1]) for t_id in target_ids]
                # negative_source_ids = [self.neg_sample_ori(source_node_ids, \
                #     ori_edges[ori_edges[:, 0] == t_id][:, 1].tolist()) for t_id in target_ids]
                sn = min([len(neg_ids) for neg_ids in negative_source_ids])
                negative_source_ids = [neg_ids[:sn] for neg_ids in negative_source_ids]

                # Q | K | K- 
                if relation_level:
                    # 这里直接将很相似的两个点进行剔除的操作，保留相应的节点数量
                    positive_source_ids = positive_source_ids + node_dict[source_type][0]
                    query_emb = node_emb[torch.LongTensor(positive_source_ids)]
                    neg_emb   = node_emb[torch.LongTensor(negative_source_ids)]
                    masks = cos(query_emb, neg_emb) < 0.999
                    sn = torch.min(torch.sum(masks, dim=1))
                    # negative_source_ids = np.array(negative_source_ids)
                    # negative_source_ids = np.array([idxs[m][:sn] for idxs, m in zip(negative_source_ids, masks.cpu().numpy())])
                    negative_source_ids = to2Darr(negative_source_ids)
                    negative_source_ids = to2Darr([idxs[m][:sn] for idxs, m in zip(negative_source_ids, masks.cpu().numpy())])

                if relation_level:
                    source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1))
                else:
                    source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1) + node_dict[source_type][0])
                emb = node_emb[source_ids]
                # print("emb shape:", emb.shape)
                if use_queue and len(self.neg_queue[source_type][relation_type]) // n_nodes > 0:
                    tmp = self.neg_queue[source_type][relation_type]
                    stx = len(tmp) // n_nodes
                    tmp = tmp[: stx * n_nodes].reshape(n_nodes, stx, -1)
                    if relation_level:
                        rep_size = sn.cpu() + 1 + stx
                    else:
                        rep_size = sn + 1 + stx
                    # rep_size = sn + 1 + stx
                    source_emb = torch.cat([emb, tmp], dim=1)
                    del tmp
                    source_emb = source_emb.reshape(n_nodes * rep_size, -1)
                else:
                    if relation_level:
                        rep_size = sn.cpu() + 1
                    else:
                        rep_size = sn + 1
                    source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)

                target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
                target_emb = node_emb[target_ids.reshape(-1)]
                res = matcher.forward(target_emb, source_emb)
                res = res.reshape(n_nodes, rep_size)
                ress += [res.detach()]
                losses += F.log_softmax(res, dim=-1)[:,0].mean()
                if update_queue and 'L1' not in relation_type and 'L2' not in relation_type:
                    tmp = self.neg_queue[source_type][relation_type]
                    self.neg_queue[source_type][relation_type] = \
                        torch.cat([node_emb[source_node_ids].detach(), tmp], dim=0)[:int(self.neg_queue_size * n_nodes)]
        return -losses / len(ress), ress

    def structure_loss(self, node_emb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False, relation_level = False):
        losses = 0
        # ress = []

        node_nbr_dict = defaultdict( # target_id
                            lambda: defaultdict( #source_type
                                lambda: defaultdict( #relation_type
                                    list #node list
                            )))


        target_node_ids = set()
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:

                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type].copy()
                if len(rem_edges) <= 8:
                    continue
                tmp_target_ids = rem_edges[:, 0] + node_dict[target_type][0]
                for _ in tmp_target_ids.tolist():
                    target_node_ids.add(_)

                for edge in rem_edges:
                    node_nbr_dict[edge[0] + node_dict[target_type][0]][source_type][relation_type].append(edge[1]+node_dict[source_type][0])

        target_node_ids = list(target_node_ids)
        # query_emb   = node_emb[torch.LongTensor(target_node_ids)]
        schema_emb  = torch.FloatTensor([]).to(self.device)

        for idx, target_id in enumerate(target_node_ids):
            tar_schema_emb = torch.FloatTensor([]).to(self.device)
            for source_type in node_nbr_dict[target_id]:
                if source_type not in self.structure_map_dict:
                    continue
                for relation_type in node_nbr_dict[target_id][source_type]:
                    structure_map = self.structure_map_dict[source_type][relation_type]
                    if relation_type not in self.structure_map_dict[source_type]:
                        continue
                    tmp_ids = np.random.choice(node_nbr_dict[target_id][source_type][relation_type], size=5).tolist()
                    tar_schema_emb = torch.cat([tar_schema_emb, structure_map(node_emb[torch.LongTensor(tmp_ids)])], dim = 0)
            schema_emb = torch.cat([schema_emb, torch.mean(tar_schema_emb, dim=0, keepdim=True)])
        # fixed original no transform data, version : V1
        # for target_id in target_node_ids:
        #     schema_ids = []
        #     for source_type in node_nbr_dict[target_id]:
        #         if source_type not in self.link_dec_dict:
        #             continue
        #         for relation_type in node_nbr_dict[target_id][source_type]:
        #             if relation_type not in self.link_dec_dict[source_type]:
        #                 continue
        #             schema_ids.extend(np.random.choice(node_nbr_dict[target_id][source_type][relation_type], size=5).tolist())
        #     # print("schema ids : ", schema_ids, "node dim shape: ", query_emb.shape[0])
        #     schema_emb = torch.cat([schema_emb, torch.mean(node_emb[torch.LongTensor(schema_ids)].detach(), dim=0, keepdim=True)], dim=0)
        
        schema_emb = self.sigm(schema_emb)
        
        schema_idxs = list()
        for idx in range(len(target_node_ids)):
            schema_idxs.append([idx] + [_ for _ in range(idx)] + [_ for _ in range(idx + 1, len(target_node_ids))])
        query_schema_emb = schema_emb[schema_idxs]

        # for _ in range(len(target_node_ids)):
        tmp = torch.unsqueeze(self.neg_structure_queue, 0)
        # print("tmp shape : ", tmp.shape, " query shema shape : ", query_schema_emb.shape[0])
        tmp = tmp.repeat(query_schema_emb.shape[0], 1, 1)
        query_schema_emb = torch.cat([query_schema_emb, tmp], dim = 1)
        del tmp

        self.neg_structure_queue = torch.cat([schema_emb.detach(), self.neg_structure_queue], dim=0)[:self.neg_structure_queue_size]

        rep_size = query_schema_emb.shape[1]

        query_emb = node_emb[torch.LongTensor(target_node_ids)]
        query_emb = query_emb.repeat(rep_size, 1, 1) 

        query_emb = query_emb.reshape(len(target_node_ids) * rep_size, -1)
        query_schema_emb = query_schema_emb.reshape(len(target_node_ids) * rep_size, -1)
        res = self.structure_matcher(query_emb, query_schema_emb)
        res = res.reshape(len(target_node_ids), rep_size)

        losses += F.log_softmax(res, dim=-1)[:,0].mean()

        return -losses

    def text_loss(self, reps, texts, w2v_model, device):
        def parse_text(texts, w2v_model, device):
            idxs = []
            pad  = w2v_model.wv.vocab['eos'].index
            for text in texts:
                idx = []
                for word in ['bos'] + preprocess_string(text) + ['eos']:
                    if word in w2v_model.wv.vocab:
                        idx += [w2v_model.wv.vocab[word].index]
                idxs += [idx]
            mxl = np.max([len(s) for s in idxs]) + 1
            inp_idxs = []
            out_idxs = []
            masks    = []
            for i, idx in enumerate(idxs):
                inp_idxs += [idx + [pad for _ in range(mxl - len(idx) - 1)]]
                out_idxs += [idx[1:] + [pad for _ in range(mxl - len(idx))]]
                masks    += [[1 for _ in range(len(idx))] + [0 for _ in range(mxl - len(idx) - 1)]]
            return torch.LongTensor(inp_idxs).transpose(0, 1).to(device), \
                   torch.LongTensor(out_idxs).transpose(0, 1).to(device), torch.BoolTensor(masks).transpose(0, 1).to(device)
        inp_idxs, out_idxs, masks = parse_text(texts, w2v_model, device)
        pred_prob = self.attr_decoder(inp_idxs, reps.repeat(inp_idxs.shape[0], 1, 1))      
        return self.ce(pred_prob[masks], out_idxs[masks]).mean()

    def feat_loss(self, reps, out):
        return -self.attr_decoder(reps, out).mean()


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

    
class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''

    def __init__(self, n_hid, n_out, temperature = 0.1):
        super(Matcher, self).__init__()
        self.n_hid          = n_hid
        self.linear    = nn.Linear(n_hid,  n_out)
        self.sqrt_hd     = math.sqrt(n_out)
        self.drop        = nn.Dropout(0.2)
        self.cosine      = nn.CosineSimilarity(dim=1)
        self.cache       = None
        self.temperature = temperature
    def forward(self, x, ty, use_norm = True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)

class StructureMapping(nn.Module):

    def __init__(self, n_hid, n_out):
        super(StructureMapping, self).__init__()
        self.n_hid  = n_hid
        self.linear = nn.Linear(n_hid, n_out)
        self.drop   = nn.Dropout(0.2)
    
    def forward(self, x):
        return self.drop(self.linear(x))

    
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs   

    
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, n_word, ninp, nhid, nlayers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(nhid, nhid, nlayers)
        self.encoder = nn.Embedding(n_word, nhid)
        self.decoder = nn.Linear(nhid, n_word)
        self.adp     = nn.Linear(ninp + nhid, nhid)
    def forward(self, inp, hidden = None):
        emb = self.encoder(inp)
        if hidden is not None:
            emb = torch.cat((emb, hidden), dim=-1)
            emb = F.gelu(self.adp(emb))
        output, _ = self.rnn(emb)
        decoded = self.decoder(self.drop(output))
        return decoded
    def from_w2v(self, w2v):
        initrange = 0.1
        self.encoder.weight.data = w2v
        self.decoder.weight = self.encoder.weight
        
        self.encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False
        