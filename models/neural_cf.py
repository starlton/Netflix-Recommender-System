import torch 
import torch.nn as nn 

class NeuralCF(nn.Module): 
    """ 
    Neural Collaborative Filtering Model
    """

    def __init__(self, n_users, n_items, embed_dim=64): 
        super().__init__() 
        self.user_emb = nn.Embedding(n_users, embed_dim) 
        self.item_emb = nn.Embedding(n_items, embed_dim) 

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, users, items): 
        u = self.user_emb(users) 
        i = self.item_emb(items) 
        x = torch.cat([u, i], dim=1) 
        return self.mlp(x).squeeze() 
    