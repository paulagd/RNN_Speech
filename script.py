# -*- coding: utf-8 -*-
"""SPEECH.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1htQ7Gnn4rjKZlrT_5IToBlTlFo7f8XJs
"""
import datetime
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from IPython import embed
from tqdm import trange


# The Dictionary class is used to map tokens (characters, words, subwords) into consecutive integer indexes.
# The index 0 is reserved for padding sequences up to a fixed lenght, and the index 1 for any 'unknown' character
class Dictionary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        return len(self.idx2token)


def batch_generator(data, batch_size, token_size):
    """Yield elements from data in chunks with a maximum of batch_size sequences and token_size tokens."""
    minibatch, sequences_so_far, tokens_so_far = [], 0, 0
    for ex in data:
        minibatch.append(ex)
        seq_len = len(ex[0])
        if seq_len > token_size:
            ex = (ex[0][:token_size], ex[1])
            seq_len = token_size
        sequences_so_far += 1
        tokens_so_far += seq_len
        if sequences_so_far == batch_size or tokens_so_far == token_size:
            yield minibatch
            minibatch, sequences_so_far, tokens_so_far = [], 0, 0
        elif sequences_so_far > batch_size or tokens_so_far > token_size:
            yield minibatch[:-1]
            minibatch, sequences_so_far, tokens_so_far = minibatch[-1:], 1, len(minibatch[-1][0])
    if minibatch:
        yield minibatch


def pool_generator(data, batch_size, token_size, shuffle=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*token_size, sorts examples within
    each chunk, then batch these examples and shuffle the batches.
    """
    for p in batch_generator(data, batch_size * 100, token_size * 100):
        p_batch = batch_generator(sorted(p, key=lambda t: len(t[0]), reverse=True), batch_size, token_size)
        p_list = list(p_batch)
        if shuffle:
            for b in random.sample(p_list, len(p_list)):
                yield b
        else:
            for b in p_list:
                yield b


class CharRNNClassifier(torch.nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size,
                 model="lstm", num_layers=1, dropout=0, bidirectional=False, pad_idx=0, bn=False):
        super().__init__()
        self.isBN = bn
        self.model = model.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.embed = torch.nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, input_lengths):
        encoded = self.embed(input)
        # if self.isBN:
        #     F.dropout(encoded, self.dropout, training=self.training, inplace=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths)
        output, hidden = self.rnn(packed)
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=float('-inf'))
        output = F.adaptive_max_pool1d(padded.permute(1, 2, 0), 1).view(-1, self.hidden_size)

        if self.isBN:
            output = self.bn(output)
            # print("batchnorm!")
        else:
            F.dropout(output, self.dropout, training=self.training, inplace=True)
        output = self.h2o(output)
        return output


class CharCNNClassifier(torch.nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1, dropout=0, pad_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed = torch.nn.Embedding(input_size, embedding_size)
        # 7, 7, 3, 3, 3, 3
        # maxpool de 3 una si, una no
        # TENSORBORARD
        self.cnn = nn.Sequential(nn.Conv1d(embedding_size, 128, kernel_size=7), nn.ELU(), nn.MaxPool1d(3),
                                 nn.Conv1d(128, self.hidden_size, kernel_size=7), nn.ELU(),
                                 nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3), nn.ELU(), nn.MaxPool1d(3),
                                 nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3), nn.ELU(),
                                 nn.Conv1d(self.hidden_size, 512, kernel_size=3), nn.ELU(), nn.MaxPool1d(3))
        # nn.Conv1d(self.hidden_size, 512, kernel_size=3), nn.ELU(), nn.MaxPool1d(3))
        self.global_pool = torch.nn.AdaptiveAvgPool1d(output_size)
        # self.fc = nn.Linear(self.hidden_size, output_size)
        self.fc = nn.Linear(512, output_size)

    def forward(self, input, input_lengths):
        # embed()

        encoded = self.embed(input)
        encoded = encoded.permute(1, 2, 0)

        output = self.cnn(encoded)
        output = torch.mean(output, dim=-1)
        F.dropout(output, self.dropout, training=self.training, inplace=True)
        output = self.fc(output)
        # F.dropout(output, self.dropout, training=self.training, inplace=True)
        # output = self.fc2(output)
        return output


def train(writer, epoch, device, criterion, model, optimizer, data, batch_size, token_size, log=False):
    model.train()
    total_loss = 0
    nsentences = 0
    ntokens = 0
    niterations = 0
    ncorrect = 0
    for batch in pool_generator(data, batch_size, token_size, shuffle=True):
        # Get input and target sequences from batch
        X = [torch.from_numpy(d[0]) for d in batch]
        X_lengths = [x.numel() for x in X]
        ntokens += sum(X_lengths)
        X_lengths = torch.tensor(X_lengths, dtype=torch.long, device=device)
        y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
        # Pad the input sequences to create a matrix
        X = torch.nn.utils.rnn.pad_sequence(X).to(device)
        model.zero_grad()
        output = model(X, X_lengths)
        # embed()
        # Calculates accuracy
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()

        # Calculates the loss
        loss = criterion(output, y)
        total_loss += loss.item()
        nsentences += y.numel()
        loss.backward()
        optimizer.step()
        niterations += 1
        if device == 'cuda':
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/acc', 100. * ncorrect / nsentences, epoch)
            # print('[Training epoch {}: {}/{}] Loss: {}'.format(j, i, len(train_loader), loss.item()))
        else:
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/acc', 100. * ncorrect / nsentences, epoch)
            # print('[Training epoch {}: {}/{}] Loss: {}'.format(j, i, len(train_loader), loss.item()))

    total_loss = total_loss/nsentences
    dev_acc = 100. * ncorrect / nsentences
    if log:
        print(f'Train: wpb={ntokens//niterations}, bsz={nsentences//niterations}, num_updates={niterations}')
        print(model)
    return total_loss, dev_acc


def validate(writer, device, criterion, model, data, batch_size, token_size, epoch):
    model.eval()
    # calculate accuracy on validation set
    ncorrect = 0
    nsentences = 0
    total_loss = 0
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device=device)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            output = model(X, X_lengths)

            # Calculates the loss
            loss = criterion(output, y)
            total_loss += loss.item()
            # Calculates . accuracy
            ncorrect += (torch.max(output, 1)[1] == y).sum().item()
            nsentences += y.numel()

        dev_acc = 100. * ncorrect / nsentences
        total_loss = total_loss/nsentences
        writer.add_scalar('val/loss', total_loss, epoch)
        writer.add_scalar('val/acc', dev_acc, epoch)

    return total_loss, dev_acc


def main(name_logs, no_logs, submission, path, model_type, optimizer, lr, dropout, epochs, hidden_size,
         embedding_size, num_layers, bidirectional, batchnorm):
    # Input data files are available in the "../input/" directory.
    print(os.listdir(path))

    # Init random seed to get reproducible results
    seed = 1111
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Select 'GPU On' on kernel settings if desired.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    torch.cuda.manual_seed(seed)

    date = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    if no_logs:
        writer = SummaryWriter(log_dir=f'logs/no_logs/')
    else:
        # writer = SummaryWriter(log_dir=f'logs/logs_{device}_{date}/')
        writer = SummaryWriter(log_dir=f'logs/'+name_logs+'/')

    # Any results you write to the current directory are saved as output.
    x_train_full = open(path+"/x_train.txt").read().splitlines()
    y_train_full = open(path+"/y_train.txt").read().splitlines()
    # print('Example:')
    # print('LANG =', y_train_full[0])
    # print('TEXT =', x_train_full[0])
    char_vocab = Dictionary()
    pad_token = '<pad>'  # reserve index 0 for padding
    unk_token = '<unk>'  # reserve index 1 for unknown token
    pad_index = char_vocab.add_token(pad_token)
    unk_index = char_vocab.add_token(unk_token)

    # join all the training sentences in a single string
    # and obtain the list of different characters with set
    chars = set(''.join(x_train_full))
    for char in sorted(chars):
        char_vocab.add_token(char)
    print("Vocabulary:", len(char_vocab), "UTF characters")

    lang_vocab = Dictionary()
    # use python set to obtain the list of languages without repetitions
    languages = set(y_train_full)
    for lang in sorted(languages):
        lang_vocab.add_token(lang)

    print("Labels:", len(lang_vocab), "languages")

    # From token or label to index
    # print('a ->', char_vocab.token2idx['a'])
    # print('cat ->', lang_vocab.token2idx['cat'])
    # print(y_train_full[0], x_train_full[0][:10])
    x_train_idx = [np.array([char_vocab.token2idx[c] for c in line]) for line in x_train_full]
    y_train_idx = np.array([lang_vocab.token2idx[lang] for lang in y_train_full])
    # print(y_train_idx[0], x_train_idx[0][:10])

    # Radomly select 15% of the database for validation
    # Create lists of (input, target) tuples for training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train_idx, y_train_idx,
                                                      test_size=0.15, random_state=seed)
    train_data = [(x, y) for x, y in zip(x_train, y_train)]
    val_data = [(x, y) for x, y in zip(x_val, y_val)]
    print(len(train_data), "training samples")
    print(len(val_data), "validation samples")


    # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # It is useful when training a classification problem.
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size, token_size = 256, 200000
    ntokens = len(char_vocab)
    nlabels = len(lang_vocab)

    if model_type == 'cnn':
        model = CharCNNClassifier(ntokens, embedding_size, hidden_size, nlabels, num_layers=num_layers,
                                  dropout=dropout, pad_idx=pad_index).to(device)
    else:
        model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels, dropout=dropout,
                                  num_layers=num_layers, bidirectional=bidirectional, model=model_type,
                                  pad_idx=pad_index, bn=batchnorm).to(device)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f'Training cross-validation model for {epochs} epochs')

    if not submission:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(writer, epoch, device, criterion, model, optimizer, train_data,
                                          batch_size, token_size, log=epoch == 1)
            val_loss, val_acc = validate(writer, device, criterion, model, val_data, batch_size,
                                         token_size, epoch)
            print(f'| TRAINING epoch {epoch:03d} | loss={train_loss:.3f} | acc={train_acc:.1f}%')
            print(f'| VALIDATION epoch {epoch:03d} | loss={val_loss:.3f} | acc={val_acc:.1f}%')
    else:
        print(f'Training final model for {epochs} epochs')

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(writer, epoch, device, criterion, model, optimizer, train_data + val_data,
                                          batch_size, token_size, log=epoch == 1)
            print(f'| TRAINING epoch {epoch:03d} | loss={train_loss:.3f} | acc={train_acc:.1f}%')

        x_test_txt = open(path+"/x_test.txt").read().splitlines()
        x_test_idx = [np.array([char_vocab.token2idx[c] if c in char_vocab.token2idx
                      else unk_index for c in line]) for line in x_test_txt]
        test_data = [(x, idx) for idx, x in enumerate(x_test_idx)]

        index, labels = test(device, model, test_data, batch_size, token_size)
        order = np.argsort(index)
        labels = labels[order]

        with open('submission.csv', 'w') as f:
            print('Id,Language', file=f)
            for sentence_id, lang_id in enumerate(labels):
                language = lang_vocab.idx2token[lang_id]
                if sentence_id < 10:
                    print(f'{sentence_id},{language}')
                print(f'{sentence_id},{language}', file=f)


def test(device, model, data, batch_size, token_size):
    model.eval()
    sindex = []
    labels = []
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            label = torch.max(answer, 1)[1].cpu().numpy()
            # Save labels and sentences index
            labels.append(label)
            sindex += [d[1] for d in batch]
    return np.array(sindex), np.concatenate(labels)


# path = '/content/drive/My Drive/Colab_Notebooks/wili'
path = 'wili'
# path = '../input'
model_type = "lstm"
optimizer = 'adam'  # SGD
# optimizer = 'SGD'  # SGD
lr = 1e-3

epochs = 30
hidden_size = 128
embedding_size = 64
bidirectional = False
batchnorm = True
dropout = 0.5
num_layers = 1
submission = False
no_logs = False
name_logs = 'baseline_lstm_batchnorm_hz128'


main(name_logs, no_logs, submission, path, model_type, optimizer, lr, dropout, epochs, hidden_size,
     embedding_size, num_layers, bidirectional, batchnorm)

"""De esta forma podrías insertar una imagen
![nombre de la imagen][img1]

O dos, sin ensuciar tu espacio de escritura.
![nombre de la imagen2][img2]

[img1]: /ruta/a/la/imagen.jpg "Título alternativo"
[img2]: /ruta/a/la/imagen2.jpg "Título alternativo"
"""
