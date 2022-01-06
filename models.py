import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Code2Vec(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout, path_size, hidden_dim, batch_size, nb_paths):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_len = path_size
        self.nb_paths = nb_paths
        self.batch_size = batch_size

        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)

        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)

        # lstm for path encoding
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.seq_len,
            batch_first=True)
        self.hidden2path = nn.Linear(self.seq_len*self.hidden_dim, self.embedding_dim)

        # weights for fully connected layer -> vector compression
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        # attention weights
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        # dropout for prevention of coadaption of neurons
        self.do = nn.Dropout(dropout)

    def init_hidden(self):
        # the weights are of the form (nb_layers, lstm batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.seq_len, self.batch_size*self.nb_paths, self.hidden_dim)
        hidden_b = torch.randn(self.seq_len, self.batch_size*self.nb_paths, self.hidden_dim)

        #if self.hparams.on_gpu:
        #    hidden_a = hidden_a.cuda()
        #    hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, starts, paths, ends):

        self.hidden = self.init_hidden()
        
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
        lengths = []
        for l in [[[i for i in j if i != 1] for j in k] for k in paths.tolist()]:
            for m in l:
                if len(m) != 0:
                    lengths.append(len(m))
                else:
                    lengths.append(1)


        # transform input for lstm -> (lstm batch size, max path length, embedding dim)
        lstm_in = embedded_paths.view(self.batch_size*self.nb_paths, self.seq_len, -1)
        #lstm_in = [batch size * max length, max_path_length, embedding dim]
        #put each path through the LSTM

        lstm_in_nopad = torch.nn.utils.rnn.pack_padded_sequence(lstm_in, torch.FloatTensor(lengths), batch_first=True, enforce_sorted=False)

        lstm_out_pad, self.hidden = self.lstm(lstm_in_nopad, self.hidden)
        #lstm_out = [batch size * max length, max_path_length, hidden dim]
        #lstm_out, _ = self.lstm(lstm_in)
        #lstm_out = [batch size * max length, embedding dim, 1]

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_pad, batch_first=True, total_length=self.seq_len)

        encoded_path = self.hidden2path(lstm_out.contiguous().view(-1, self.seq_len*self.hidden_dim))
        #encoded_paths = [batch size * max length, embedding dim]
        encoded_paths = encoded_path.view(self.batch_size, self.nb_paths, -1)
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
