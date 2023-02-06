import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_embedding_dim,
        n_hidden,
        n_layers,
        bidirectional=False,
        dropout=0.5,
        rnn_type="elman",  # can be elman, lstm, gru
        use_glove=False
    ):
        super(RNNModel, self).__init__()
        
        self.use_glove = use_glove
        self.rnn_type = rnn_type
        self.bidirectional = True if bidirectional else False

        if rnn_type == "elman":
            self.rnn = nn.RNN(
                in_embedding_dim,
                n_hidden,
                n_layers,
                nonlinearity="tanh",
                dropout=dropout,
                bidirectional=self.bidirectional,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_embedding_dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                bidirectional=self.bidirectional,
                dropout=dropout
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_embedding_dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                bidirectional=self.bidirectional,
                dropout=dropout
            )
        else:
            # TODO: implement lstm and gru
            # self.rnn = ...
            raise NotImplementedError
        
        self.in_embedder = nn.Embedding(vocab_size, in_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        print(self.bidirectional)
        self.num_directions = 1 if not self.bidirectional else 2
        print("num_directions: {}".format(self.num_directions))
        self.pooling = nn.Linear(n_hidden * self.num_directions, vocab_size)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        if not self.use_glove:
            self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.in_embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.pooling.bias)
        nn.init.uniform_(self.pooling.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.in_embedder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        pooled = self.pooling(output)
        pooled = pooled.view(-1, self.vocab_size)
        return F.log_softmax(pooled, dim=1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == "lstm":
            return weight.new_zeros(self.n_layers * self.num_directions, batch_size, self.n_hidden), weight.new_zeros(self.n_layers * self.num_directions, batch_size, self.n_hidden)
        else:
            return weight.new_zeros(self.n_layers * self.num_directions, batch_size, self.n_hidden)

    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = torch.load(f)
            model.rnn.flatten_parameters()
            return model
