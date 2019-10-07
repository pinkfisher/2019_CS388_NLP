from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
import numpy as np
import time

"""
    RNN Model for sentiment analysis
"""


def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_accuracy(preds, y):
    _, rounded_preds = torch.max(torch.sigmoid(preds), 1)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def form_input(sentence, word_vectors: WordEmbeddings, average=True):
    if average:
        sum_embedding = 0
        for word in sentence:
            sum_embedding += word_vectors.get_embedding_from_index(int(word))
        rst = sum_embedding/len(sentence)
        rst = torch.from_numpy(rst).float()
        return rst
    else:
        embeddings = [torch.from_numpy(word_vectors.get_embedding_from_index(int(word))) for word in sentence]
        return torch.stack(embeddings)


def prepare_dataset_for_RNN(train_exs: List[SentimentExample], word_vectors, pad_size=60, num_data=None):
    """

    :param train_exs:
    :param word_vectors:
    :param pad_size:
    :param num_data:
    :return: [(sentence(embedded), sentence_length), label]
    """
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = pad_size
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])

    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])

    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # Select a subset of data
    if num_data:
        train_mat = train_mat[:num_data, :]
        train_labels_arr = train_labels_arr[:num_data]

    train_data = []
    for idx in range(train_mat.shape[0]):
        x = form_input(train_mat[idx], word_vectors, average=False)
        y = train_labels_arr[idx]
        y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        train_data.append(((x, train_seq_lens[idx]), y))
    #print("number of 1s: {}".format(np.sum(train_labels_arr)))
    return train_data


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True, bidirectional=self.bidirectional,
                            dropout=0.5)

        # Readout layer
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, x_length = data

        x = x.float()

        x = self.dropout(x)

        # Sort and pack the input
        x_length, perm_idx = x_length.sort(0, descending=True)
        x = x[perm_idx]
        x = nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=True)

        # Double the number of layer if using bidirectional LSTM
        num_layer = self.layer_dim
        if self.bidirectional:
            num_layer *= 2

        packed_output, (hn, cn) = self.lstm(x)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # hidden = [batch size, hid dim * num directions]
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

    def evaluate(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        self.eval()

        with torch.no_grad():
            for (x, y) in iterator:

                predictions = self(x).squeeze(1)

                loss = criterion(predictions, y)

                acc = get_accuracy(predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def predict(self, test_exs: List[SentimentExample], word_vectors):
        test_data = prepare_dataset_for_RNN(test_exs, word_vectors)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=1,
                                                  shuffle=False)
        predictions = []
        for i, (x, y) in enumerate(test_loader):
            exs = test_exs[i]
            outputs = self.forward(x)
            _, predicted = torch.max(torch.round(torch.sigmoid(outputs)), 1)
            predictions.append(SentimentExample(exs.indexed_words, predicted[0].item()))
        return predictions


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    embedding_size = word_vectors.get_embedding_length()
    num_classes = 2
    hidden_size = 256
    batch_size = 32
    num_hidden_layer = 2
    learning_rate = 0.01

    train_data = prepare_dataset_for_RNN(train_exs, word_vectors, num_data=0)
    dev_data = prepare_dataset_for_RNN(dev_exs, word_vectors)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dataset=dev_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    num_epochs = 20
    model = LSTMModel(embedding_size, hidden_size, num_hidden_layer, num_classes, bidirectional=True)
    model = model.float()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    best_dev_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        for (x, y) in train_loader:

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(x)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, y)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # evaluate for each epoch
        train_loss, train_acc = model.evaluate(train_loader, criterion)
        dev_loss, dev_acc = model.evaluate(dev_loader, criterion)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            test_exs_predicted = model.predict(test_exs, word_vectors)

            write_sentiment_examples(test_exs_predicted, 'test-blind.output.txt', word_vectors.word_indexer)
            print("======== Finish writing output ========")

    return test_exs_predicted