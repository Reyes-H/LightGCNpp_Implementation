import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import os
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20


class LightGCNsharp(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(LightGCNsharp, self).__init__(conf, training_set, valid_set, test_set)
        self.n_layers = conf.n_layer
        self.init = conf.init
        self.warm_up = conf.warm_up
        self.lr_param = conf.lr_param
        self.ts_param = conf.ts_param
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.init)
        
        self.config_name = f'{conf.dataset}_{conf.model_name}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_nl{self.n_layers}_init{self.init}_warm{self.warm_up}_lrp{self.lr_param}_tsp{self.ts_param}'
        print()
        print(self.config_name)
        
        if os.path.exists(f'logs/{self.config_name}.txt'):
            print('Exists.')
            exit(0)
    
    def train(self):
        model = self.model.cuda()
        
        optimizer_emb = torch.optim.Adam(
            model.embedding_dict.parameters(),
            lr=self.lRate
        )

        optimizer_params = torch.optim.Adam(
            [model.alpha, model.beta, model.gamma],
            lr=self.lRate * self.lr_param
        )
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                
                # Update Embeddings
                rec_user_emb, rec_item_emb = model(detach_params=True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss_emb = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size
                
                optimizer_emb.zero_grad()
                batch_loss_emb.backward()
                optimizer_emb.step()
                
                if epoch >= self.warm_up and n % self.ts_param == 0:
                    # Update Parameters
                    rec_user_emb, rec_item_emb = model(detach_emb=True)
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    batch_loss_params = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size

                    optimizer_params.zero_grad()
                    batch_loss_params.backward()
                    torch.nn.utils.clip_grad_norm_([model.alpha, model.beta, model.gamma], max_norm=1.0)
                    optimizer_params.step()
                
                if (n+1) % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n+1, 'batch_loss:', batch_loss_emb.item())
            
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            self.evaluate(self.test('valid'), 'valid')
            result_valid = [r[:-1] for r in self.result]
            self.evaluate(self.test('test'), 'test')
            result_test = [r[:-1] for r in self.result]
            
            for _i in range(3):
                print('Valid\t', result_valid[_i*5], result_valid[_i*5+3], result_valid[_i*5+4])
            for _i in range(3):
                print('Test\t', result_test[_i*5], result_test[_i*5+3], result_test[_i*5+4])
                
            print(f'{model.alpha.item()}\t{model.beta.item()}\t{model.gamma.item()}')
            
            with open(f'logs/{self.config_name}.txt', 'a') as f:
                valid_log, test_log = '', ''
                for _i in range(3):
                    recall = result_valid[_i*5+3].split(':')[1]
                    ndcg = result_valid[_i*5+4].split(':')[1]
                    valid_log += f',{recall},{ndcg}'
                    
                    recall = result_test[_i*5+3].split(':')[1]
                    ndcg = result_test[_i*5+4].split(':')[1]
                    test_log += f',{recall},{ndcg}'
                f.write(f'{epoch+1},valid,{valid_log}\n')
                f.write(f'{epoch+1},test,{test_log}\n')
                
            with open(f'logsP/{self.config_name}.txt', 'a') as f:
                f.write(f'{model.alpha.item()}\t{model.beta.item()}\t{model.gamma.item()}\n')
                
            ndcg_valid = float(result_valid[9].split(':')[1])
            if ndcg_valid > best_valid:
                best_valid = ndcg_valid
                self.best_user_emb = self.model.embedding_dict['user_emb'].detach().cpu()
                self.best_item_emb = self.model.embedding_dict['item_emb'].detach().cpu()
                wait_cnt = 0
            else:
                wait_cnt += 1
                print(f'Patience... {wait_cnt}/{patience}')
                
            if wait_cnt == patience:
                print('Early Stopping!')
                break
            print()



    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, init):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        
        if init == 0:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.gamma = nn.Parameter(torch.tensor(1.0 / (self.layers+1)))
        elif init == 1:
            self.alpha = nn.Parameter(torch.tensor(0.6))
            self.beta = nn.Parameter(torch.tensor(-0.1))
            self.gamma = nn.Parameter(torch.tensor(0.1))
            
        self.embedding_dict = self._init_model()
        
        coo = self.data.ui_adj.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        indices = torch.stack([row, col])
        
        self.ui_adj = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).cuda()
        self.rowsum = torch.sparse.sum(self.ui_adj, dim=1).to_dense()
        
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict
        

    
    def forward(self, detach_emb=False, detach_params=False):
        if detach_emb:
            ego_embeddings = torch.cat([self.embedding_dict['user_emb'].detach(), self.embedding_dict['item_emb'].detach()], 0)
        else:
            ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
            
        if detach_params:
            alpha, beta, gamma = self.alpha.detach(), self.beta.detach(), self.gamma.detach()
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
            
            
        d_inv_left = torch.pow(self.rowsum, -alpha)
        d_inv_left = torch.where(torch.isinf(d_inv_left), torch.zeros_like(d_inv_left), d_inv_left)
        d_inv_left = d_inv_left.unsqueeze(1)  

        d_inv_right = torch.pow(self.rowsum, -beta)
        d_inv_right = torch.where(torch.isinf(d_inv_right), torch.zeros_like(d_inv_right), d_inv_right)
        d_inv_right = d_inv_right.unsqueeze(1) 
        
            
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            norm = torch.norm(ego_embeddings, dim=1) + 1e-12
            ego_embeddings = ego_embeddings / norm[:,None]
            
            embeddings_right = ego_embeddings * d_inv_right
            embeddings_adj = torch.sparse.mm(self.ui_adj, embeddings_right)
            ego_embeddings = embeddings_adj * d_inv_left
            
            all_embeddings += [ego_embeddings]
            
        embs_zero = all_embeddings[0]
        embs_prop = torch.mean(torch.stack(all_embeddings[1:], dim=1), dim=1)

        light_out = (gamma * embs_zero) + ((1 - gamma) * embs_prop)

        user_all_embeddings = light_out[:self.data.user_num]
        item_all_embeddings = light_out[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

