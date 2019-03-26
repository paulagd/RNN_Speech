import random
import numpy as np  # linear algebra
# import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import os
from sklearn.model_selection import train_test_split
from utils import pool_generator, Dictionary
from model import CharRNNClassifier


def train(model, optimizer, data, batch_size, token_size, log=False):
    model.train()
    total_loss = 0
    nsentences = 0
    ntokens = 0
    niterations = 0
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
        loss = criterion(output, y)
        total_loss += loss.item()
        nsentences += y.numel()
        loss.backward()
        optimizer.step()
        niterations += 1

    total_loss = total_loss/nsentences
    if log:
        print(f'Train: wpb={ntokens//niterations}, bsz={nsentences//niterations}, num_updates={niterations}')
    return total_loss


def validate(model, data, batch_size, token_size):
    model.eval()
    # calculate accuracy on validation set
    ncorrect = 0
    nsentences = 0
    with torch.no_grad():
        for batch in pool_generator(data, batch_size, token_size):
            # Get input and target sequences from batch
            X = [torch.from_numpy(d[0]) for d in batch]
            X_lengths = torch.tensor([x.numel() for x in X], dtype=torch.long, device=device)
            y = torch.tensor([d[1] for d in batch], dtype=torch.long, device=device)
            # Pad the input sequences to create a matrix
            X = torch.nn.utils.rnn.pad_sequence(X).to(device)
            answer = model(X, X_lengths)
            ncorrect += (torch.max(answer, 1)[1] == y).sum().item()
            nsentences += y.numel()
        dev_acc = 100. * ncorrect / nsentences
    return dev_acc


if __name__ == '__main__':

    # Input data files are available in the "../input/" directory.
    # path = '../input'
    path = 'wili'
    print(os.listdir(path))

    # Init random seed to get reproducible results
    seed = 1111
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Any results you write to the current directory are saved as output.
    x_train_full = open(path+"/x_train.txt").read().splitlines()
    y_train_full = open(path+"/y_train.txt").read().splitlines()
    print('Example:')
    print('LANG =', y_train_full[0])
    print('TEXT =', x_train_full[0])
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
    print('a ->', char_vocab.token2idx['a'])
    print('cat ->', lang_vocab.token2idx['cat'])
    print(y_train_full[0], x_train_full[0][:10])
    x_train_idx = [np.array([char_vocab.token2idx[c] for c in line]) for line in x_train_full]
    y_train_idx = np.array([lang_vocab.token2idx[lang] for lang in y_train_full])
    print(y_train_idx[0], x_train_idx[0][:10])

    # Radomly select 15% of the database for validation
    # Create lists of (input, target) tuples for training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train_idx, y_train_idx, test_size=0.15, random_state=seed)
    train_data = [(x, y) for x, y in zip(x_train, y_train)]
    val_data = [(x, y) for x, y in zip(x_val, y_val)]
    print(len(train_data), "training samples")
    print(len(val_data), "validation samples")

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Select 'GPU On' on kernel settings if desired.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    torch.cuda.manual_seed(seed)

    # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # It is useful when training a classification problem.
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    hidden_size = 256
    embedding_size = 32
    bidirectional = False
    ntokens = len(char_vocab)
    nlabels = len(lang_vocab)

    model = CharRNNClassifier(ntokens, embedding_size, hidden_size, nlabels,
                              bidirectional=bidirectional, pad_idx=pad_index).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size, token_size = 256, 200000
    epochs = 25
    print(f'Training cross-validation model for {epochs} epochs')
    for epoch in range(1, epochs + 1):
        print(f'| epoch {epoch:03d} | loss={train(model, optimizer, train_data, batch_size, token_size, log=epoch==1):.3f}')
        print(f'| epoch {epoch:03d} | valid accuracy={validate(model, val_data, batch_size, token_size):.1f}%')
