import torch
import torch.nn as nn
import torch.nn.functional as F

class Code2Vec(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout, path_size, hidden_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)

        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)

        # lstm for path encoding
        self.lstm = nn.LSTM(path_size, hidden_dim, batch_first=True)
        self.hidden2path = nn.Linear(hidden_dim, 1)

        # weights for fully connected layer -> vector compression
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        # attention weights
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        # dropout for prevention of coadaption of neurons
        self.do = nn.Dropout(dropout)
        
    def forward(self, starts, paths, ends):
        
        #starts = ends = [batch size, max length]
        #paths = [batch size, max length, max path length]
        
        W = self.W.repeat(starts.shape[0], 1, 1)
        #W = [batch size, embedding dim, embedding dim * 3]
        
        embedded_starts = self.node_embedding(starts)
        embedded_ends = self.node_embedding(ends)
        #embedded_starts = embedded_ends = [batch size, max length, embedding dim]


        # PATH ENCODING WITH LSTM
        # 1. EMBED PATHS
        # 2. FEED IN LSTM
        # -> PATH ENCODING

        embedded_paths = self.path_embedding(paths)
        #embedded_paths = [batch size, max length, max_path_length, embedding dim]

        lstm_in = embedded_paths.view(paths.size()[0]*paths.size()[1], self.embedding_dim, -1)
        #lstm_in = [batch size * max length, embedding dim, max_path_length]

        lstm_out, _ = self.lstm(lstm_in)
        #lstm_out = [batch size * max length, embedding dim, 1]

        encoded_path = self.hidden2path(lstm_out.view(paths.size()[0]*paths.size()[1], self.embedding_dim, -1))
        encoded_paths = encoded_path.view(paths.size()[0], paths.size()[1], -1)
        #encoded_paths = [batch size, max length, embedding dim]


        # CONCAT INPUT VALUES TO VECTORS

        c = self.do(torch.cat((embedded_starts, encoded_paths, embedded_ends), dim=2))
        #c = [batch size, max length, embedding dim * 3]


        # FULLY CONNECTED LAYER -> Input Vectors to Single Values
        
        c = c.permute(0, 2, 1)
        #c = [batch size, embedding dim * 3, max length]

        x = torch.tanh(torch.bmm(W, c))
        #x = [batch size, embedding dim, max length]


        # COMBINED CONTEXT VECTORS (x) & ATTENTION WEIGHTS (a) -> CODE VECTOR (z)
        
        x = x.permute(0, 2, 1)
        #x = [batch size, max length, embedding dim]
        
        a = self.a.repeat(starts.shape[0], 1, 1)
        #a = [batch size, embedding dim, 1]

        z = torch.bmm(x, a).squeeze(2)
        #z = [batch size, max length]


        #  SOFTMAX PREDICTION

        z = F.softmax(z, dim=1)
        #z = [batch size, max length]
        
        z = z.unsqueeze(2)
        #z = [batch size, max length, 1]
        
        x = x.permute(0, 2, 1)
        #x = [batch size, embedding dim, max length]
        
        v = torch.bmm(x, z).squeeze(2)
        #v = [batch size, embedding dim]
        
        out = self.out(v)
        #out = [batch size, output dim]

        return out
