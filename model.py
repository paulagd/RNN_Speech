import torch
import torch.nn.functional as F


class CharRNNClassifier(torch.nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                 model="lstm", num_layers=1, bidirectional=False, pad_idx=0):
        super().__init__()
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, input_lengths):
        encoded = self.embed(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths)
        output, hidden = self.rnn(packed)
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=float('-inf'))
        output = F.adaptive_max_pool1d(padded.permute(1, 2, 0), 1).view(-1, self.hidden_size)
        output = self.h2o(output)
        return output
