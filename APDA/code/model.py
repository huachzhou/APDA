import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
from torch.nn import Parameter
from torch.autograd import Function

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class APDA(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(APDA, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['APDA_n_layers']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')
        self.dropout = nn.Dropout(p=0.4, inplace=True)
        self.f = nn.Sigmoid()
        self.residual_coff = self.config['residual_coff']
        self.model_type = self.config['model_type']
        self.exp_coff = self.config['exp_coff']
        self.exp_on = self.config['exp_on']
        self.Graph = self.dataset.getSparseGraph()

    def computer(self, users, pos_items, neg_items, epoch):

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb])
        initial_emb = nn.functional.normalize(all_emb) 
        embs = [all_emb] 


        for layer in range(self.n_layers):
            if self.model_type != 'APDA_RC':
                all_emb = all_emb + self.residual_coff * initial_emb
            if self.model_type != 'APDA_AW':    
                all_emb = nn.functional.normalize(all_emb)
            neighbor_emb = self.edge_weight_calc(self.Graph, all_emb)
            if self.model_type != 'APDA_RC':
                all_emb = neighbor_emb + self.residual_coff * (all_emb - initial_emb) # preserve last layer embedding
            else:
                all_emb = neighbor_emb
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1) 
        users, items = torch.split(light_out, [self.num_users, self.num_items]) 
 
        return users, items, 0

    def edge_weight_calc(self, adj, emb):
        if self.model_type != 'APDA_AW':
            indices = adj._indices()
            values = adj._values()
            if self.training:
                sign = torch.sign(emb)
                random_noise = nn.functional.normalize(torch.rand(emb.shape).to(world.device)) * 0.1
                emb = emb + sign * random_noise

            start_emb = (emb[indices[0]]).detach()
            end_emb = (emb[indices[1]])
            cross_product = (torch.mul(start_emb, end_emb).mean(dim=1))  
            if self.exp_on:
                mat = 1/2 * torch.exp((2 - 2 * cross_product)/self.exp_coff) * torch.nn.functional.softplus((2 - 2 * cross_product)/self.exp_coff)
            else:
                mat = (2 - 2 * cross_product)/self.exp_coff
        if self.model_type != 'APDA_WS':
            mat = mat * values
 
        new_indices = indices[0].unsqueeze(1).expand(end_emb.shape)
        mat = torch.mul(self.dropout(end_emb), mat.unsqueeze(1).expand(end_emb.shape))

        update_all_emb = torch.zeros(emb.shape).to(world.device)
        update_all_emb.scatter_add_(0, new_indices, mat)

        return update_all_emb

    def getUsersRating(self, users, epoch):
        all_users, all_items, _ = self.computer(users, None, None, epoch)
        users_emb = all_users[users]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items, epoch):
        all_users, all_items, edge_loss = self.computer(users, pos_items, neg_items, epoch)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]


        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, edge_loss
    def bpr_loss(self, users, pos, neg, epoch):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0, edge_loss) = self.getEmbedding(users.long(), pos.long(), neg.long(), epoch)
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users)) + edge_loss*10

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        torch.cuda.empty_cache()
        return loss, reg_loss
 