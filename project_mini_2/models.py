# models.py

from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random

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

class FFNN(nn.Module):
    def __init__(self, inp, hid, out):
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        return self.softmax(self.W(self.g(self.V(x))))

    def evaluate(self, train_data):
        train_correct = 0
        for idx in range(0, len(train_data)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            x, y = train_data[idx]
            probs = self.forward(x)
            prediction = torch.argmax(probs)
            target = torch.argmax(y)
            if target == prediction:
                train_correct += 1

        print("Correct: %d, Accuracy: %f" % (train_correct, train_correct / len(train_data)))
        return train_correct


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


def prepare_dataset(train_exs: List[SentimentExample], word_vectors, pad_size=60, num_data=None):
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
        x = form_input(train_mat[idx], word_vectors)
        y = train_labels_arr[idx]
        y_onehot = torch.zeros(2)
        y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
        train_data.append((x, y_onehot))
    print("number of 1s: {}".format(np.sum(train_labels_arr)))
    return train_data


# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample],
                        word_vectors: WordEmbeddings) -> List[SentimentExample]:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    predictions on the *blind* test examples passed in. Returned predictions should be SentimentExample objects with
    predicted labels and the same sentences as input (but these won't be read by the external code). The code is set
    up to go all the way to test predictions so you have more freedom to handle example processing as you see fit.
    :param train_exs:
    :param dev_exs:
    :param test_exs:
    :param word_vectors:
    :return:
    """

    embedding_size = word_vectors.get_embedding_length()
    num_classes = 2
    hidden_size = 64

    train_data = prepare_dataset(train_exs, word_vectors)
    dev_data = prepare_dataset(dev_exs, word_vectors)
    test_data = prepare_dataset(test_exs, word_vectors)


    num_epochs = 20
    num_data = len(train_data)
    ffnn = FFNN(embedding_size, hidden_size, num_classes)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ffnn.parameters(), lr=0.1)

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(num_data)]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x, y = train_data[idx]
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            loss = torch.neg(torch.log(probs)).dot(y)
            #loss = criterion(probs, y)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print("Epoch %d: Total loss: %f" %(epoch, total_loss))
        print("Dev: ", end=" ")
        ffnn.evaluate(dev_data)
        print("Train:", end=" ")
        ffnn.evaluate(train_data)

    # Make prediction
    predictions = []
    for idx in range(0, len(test_data)):
        x, y = test_data[idx]
        exs = test_exs[idx]
        probs = ffnn.forward(x)
        predictions.append(SentimentExample(exs.indexed_words, torch.argmax(probs).item()))
    return predictions

def prepare_dataset_for_RNN(train_exs: List[SentimentExample], word_vectors, pad_size=60, num_data=None):
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
    print("number of 1s: {}".format(np.sum(train_labels_arr)))
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=self.bidirectional)

        # Readout layer
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, x_length = x
        num_data = x.size(0)
        x = x.float()
        x_length, perm_idx = x_length.sort(0, descending=True)
        x = x[perm_idx]
        x = nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=True)

        num_layer = self.layer_dim
        if self.bidirectional:
            num_layer *= 2

        # Initialize hidden state with zeros
        h0 = torch.zeros(num_layer, num_data, self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(num_layer, num_data, self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        packed_output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        #output = self.fc(output[:, -1, :])

        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)  # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)

    def evaluate(self, data_loader):
        correct = 0
        total = 0

        for x, y in data_loader:
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()

        accuracy = 100 * correct / total
        return accuracy


# Analogous to train_ffnn, but trains your fancier model.
def train_evaluate_fancy(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    embedding_size = word_vectors.get_embedding_length()
    num_classes = 2
    hidden_size = 256
    batch_size = 10
    num_hidden_layer = 1
    learning_rate = 0.1

    train_data = prepare_dataset_for_RNN(train_exs, word_vectors, num_data=0)
    dev_data = prepare_dataset_for_RNN(dev_exs, word_vectors)
    test_data = prepare_dataset_for_RNN(test_exs, word_vectors)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dataset=dev_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    num_epochs = 50
    model = LSTMModel(embedding_size, hidden_size, num_hidden_layer, num_classes, bidirectional=True)
    model = model.float()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    iter = 0
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):

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

            iter += 1

        # evaluate for each epoch
        train_accuracy = model.evaluate(train_loader)
        dev_accuracy = model.evaluate(dev_loader)
        print('Epoch: {}, Iteration: {}. Loss: {}. Train accuracy: {}, Dev accuracy {}.'
              .format(epoch, iter, loss.item(), train_accuracy, dev_accuracy))

    train_accuracy = model.evaluate(train_loader)
    dev_accuracy = model.evaluate(dev_loader)
    print('Final result: Train accuracy: {}, Dev accuracy {}.'
          .format(train_accuracy, dev_accuracy))

    # Make prediction
    predictions = []
    for idx in range(0, len(test_data)):
        x, y = test_data[idx]
        x, _ = x
        x = x.unsqueeze(0) # change to [1, seq_len, embedding_size]
        exs = test_exs[idx]
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(SentimentExample(exs.indexed_words, predicted[0].item()))
    return predictions