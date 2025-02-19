import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import pickle as pkl
import os
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20


class LightGCNpp(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(LightGCNpp, self).__init__(conf, training_set, valid_set, test_set)
        self.n_layers = conf.n_layer
        self.alpha = conf.alpha
        self.beta = conf.beta
        self.gamma = conf.gamma
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.alpha, self.beta, self.gamma)
        
        self.config_name = f'{conf.dataset}_{conf.model_name}_seed{conf.seed}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_nl{self.n_layers}_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}'
        print()
        print(self.config_name)

        self.save = conf.save
        
        if os.path.exists(f'logs/{self.config_name}.txt'):
            print('Exists.')
            if not self.save:
                exit(0)
            else:
                print('But save exception...')

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
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

            if not self.save:
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
                
            ndcg_valid = float(result_valid[9].split(':')[1])
            if ndcg_valid > best_valid:
                best_valid = ndcg_valid
                self.best_user_emb = self.model.embedding_dict['user_emb'].detach().cpu()
                self.best_item_emb = self.model.embedding_dict['item_emb'].detach().cpu()
                wait_cnt = 0

                if self.save:
                    self.user_emb = self.user_emb.detach().cpu()
                    self.item_emb = self.item_emb.detach().cpu()
                    with open(f'embs/{self.config_name}.pkl', 'wb') as f:
                        pkl.dump([self.user_emb, self.item_emb, self.data.user, self.data.item], f)
                        
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
    def __init__(self, data, emb_size, n_layers, alpha, beta, gamma):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.norm_adj = data.normalize_graph_mat(data.ui_adj, self.alpha, self.beta)
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            norm = torch.norm(ego_embeddings, dim=1) + 1e-12
            ego_embeddings = ego_embeddings / norm[:,None]
            
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
            
        embs_zero = all_embeddings[0]
        embs_prop = torch.mean(torch.stack(all_embeddings[1:], dim=1), dim=1)

        light_out = (self.gamma * embs_zero) + ((1 - self.gamma) * embs_prop)

        user_all_embeddings = light_out[:self.data.user_num]
        item_all_embeddings = light_out[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


