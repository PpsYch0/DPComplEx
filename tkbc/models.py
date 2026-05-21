# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import math
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import pickle
import random
class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)#,x,y
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"
                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        # for i in range(250):
        #     print(queries[i],ranks[i])
        return ranks #, x,y

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    all_ts_ids = torch.arange(0, scores.shape[1]).cuda()[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class TPComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TPComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),
            nn.Embedding(sizes[1], 2 * rank, sparse=True),
            nn.Embedding(sizes[2], 6 * rank, sparse=True)
            
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank]+time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+time[:, 3*self.rank:4*self.rank]
        rhs = rhs[:, :self.rank]+time[:, 4*self.rank:5*self.rank], rhs[:, self.rank:]+time[:, 5*self.rank:6*self.rank]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank]
        
        time = time[:, :self.rank], time[:, self.rank:2*self.rank]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        lhs = lhs[:, :self.rank]+time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+time[:, 3*self.rank:4*self.rank]
        rhs = rhs[:, :self.rank]+time[:, 4*self.rank:5*self.rank], rhs[:, self.rank:]+time[:, 5*self.rank:6*self.rank]
        
        ##
        bias_t_r = time[:, 4*self.rank:5*self.rank]
        bias_t_i = time[:, 5*self.rank:6*self.rank]
        ##
        
        
        time = time[:, :self.rank], time[:, self.rank:2*self.rank]
        

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + torch.sum((lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * bias_t_r,1,keepdim=True) +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t() + torch.sum((lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * bias_t_i,1,keepdim=True)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x): 
        raise NotImplementedError("no.")
        
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank]+time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+time[:, 3*self.rank:4*self.rank]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank]
        time = time[:, :self.rank], time[:, self.rank:2*self.rank]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)
    
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath,fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])
        
class testmodel(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,args, batch_size : int = 1000 ,
            no_time_emb=False, init_size: float = 1e-2 
    ):
        super(testmodel, self).__init__()
        
        self.sizes = sizes
        self.rank = rank
        self.args = args
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True), #7128 * 1594
            nn.Embedding(sizes[1], 2 * rank, sparse=True), #460 * 1594
            nn.Embedding(sizes[2], 6 * rank, sparse=True), #7128 * 1594
            
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.no_time_emb = no_time_emb
        self.x = 0
        self.y = 0
        # s = {'gdelt':0.03459657450174849 ,'ICEWS14':0.43876899348106374,'yago15k':0.21857639060690925 ,'ICEWS05-15': 0.4756332116411938} 
        # o = {'gdelt':0.032808231257024346,'ICEWS14':0.3618899015096552,'yago15k':0.0034682573227794436,'ICEWS05-15': 0.4052256427716335}
        # self.s = s[self.args.dataset]
        # self.o = o[self.args.dataset]
        self.s,self.o = self.load_quadruples('tkbc/data/'+self.args.dataset+'/', 'train.pickle')
        print(self.s,'\n',self.o)
        self.batch_size = batch_size
        current_seed = torch.initial_seed() if torch.cuda.is_available() else torch.initial_seed()
        with open('result/'+args.dataset+'.txt', 'a') as f:  
            f.write(f"seed:{current_seed}\n")

    def load_quadruples(self,inPath, fileName):
        pass
    
    def Gate(self, x):
        pass

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        lhs = lhs[:, :self.rank]+ self.x * time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+ self.y * time[:, 3*self.rank:4*self.rank]
        rhs = rhs[:, :self.rank]+ self.x * time[:, 4*self.rank:5*self.rank], rhs[:, self.rank:]+ self.y * time[:, 5*self.rank:6*self.rank]
        # lhs = lhs[:, :self.rank]+ self.gs * time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+ self.gs * time[:, 3*self.rank:4*self.rank]
        # rhs = rhs[:, :self.rank]+ self.go * time[:, 4*self.rank:5*self.rank], rhs[:, self.rank:]+ self.go * time[:, 5*self.rank:6*self.rank]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank]
        
        time = time[:, :self.rank], time[:, self.rank:2*self.rank]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])  #(1000,3188)
        rel = self.embeddings[1](x[:, 1])  #(1000,3188)
        rhs = self.embeddings[0](x[:, 2])  #(1000,3188)
        time = self.embeddings[2](x[:, 3]) #(1000,9564)
        self.Gate(x)
        lhs = lhs[:, :self.rank]+ self.x * time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+ self.y * time[:, 3*self.rank:4*self.rank]
        rhs = rhs[:, :self.rank]+ self.x * time[:, 4*self.rank:5*self.rank], rhs[:, self.rank:]+ self.y * time[:, 5*self.rank:6*self.rank]
        bias_t_r = self.y * time[:, 4*self.rank:5*self.rank]
        bias_t_i = self.y * time[:, 5*self.rank:6*self.rank]
        # lhs = lhs[:, :self.rank]+ self.gs * time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+ self.gs * time[:, 3*self.rank:4*self.rank]
        # rhs = rhs[:, :self.rank]+ self.go * time[:, 4*self.rank:5*self.rank], rhs[:, self.rank:]+ self.go * time[:, 5*self.rank:6*self.rank]
        # bias_t_r = self.go * time[:, 4*self.rank:5*self.rank]
        # bias_t_i = self.go * time[:, 5*self.rank:6*self.rank]
        ##
        
        
        time = time[:, :self.rank], time[:, self.rank:2*self.rank]
        

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + torch.sum((lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * bias_t_r,1,keepdim=True) +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t() + torch.sum((lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * bias_t_i,1,keepdim=True)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x): 
        raise NotImplementedError("no.")
        
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)


    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank]+ self.x * time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+ self.x * time[:, 3*self.rank:4*self.rank]
        # lhs = lhs[:, :self.rank]+ self.gs * time[:, 2*self.rank:3*self.rank], lhs[:, self.rank:]+ self.gs * time[:, 3*self.rank:4*self.rank]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank]
        time = time[:, :self.rank], time[:, self.rank:2*self.rank]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)