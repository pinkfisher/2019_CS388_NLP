# models.py

from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
import numpy as np
import random
import time


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


def prepare_dataset(train_exs: List[SentimentExample], word_vectors, pad_size=60, num_data=None):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = pad_size
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])

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
        y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        x = x.unsqueeze(0)  # size change to [1, 2] and [1, 1] for CrossEntropyLoss
        y = y.unsqueeze(0)
        #y_onehot = torch.zeros(2)
        #y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
        train_data.append((x, y))
    print("number of 1s: {}".format(np.sum(train_labels_arr)))
    return train_data


class FFNN(nn.Module):
    def __init__(self, inp, hid, out, num_layer=1, activation="Tanh"):
        super(FFNN, self).__init__()

        self.num_layer = num_layer

        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        #self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

        if num_layer == 0:
            self.V0 = nn.Linear(inp, out)

        if num_layer == 2:
            self.W = nn.Linear(hid, hid)
            self.g1 = nn.Tanh()
            self.W1 = nn.Linear(hid, out)

        if activation == "ReLU":
            self.g = nn.ReLU()

        if activation == "LogSigmoid":
            self.g = nn.LogSigmoid()

        if activation == "RReLU":
            self.g = nn.RReLU()

        if activation == "Softplus":
            self.g = nn.Softplus()

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        #return self.softmax(self.W(self.g(self.V(x))))
        if self.num_layer == 0:
            return self.V0(x)
        if self.num_layer == 1:
            return self.W(self.g(self.V(x)))
        if self.num_layer == 2:
            return self.W1(self.g1(self.W(self.g(self.V(x)))))


    def evaluate(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        # Sets the module in evaluation mode
        self.eval()

        # Disable gradient calculation
        with torch.no_grad():
            for (x, y) in iterator:
                predictions = self(x)

                loss = criterion(predictions, y)

                acc = get_accuracy(predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def predict(self, test_exs: List[SentimentExample], word_vectors):
        test_data = prepare_dataset(test_exs, word_vectors)

        predictions = []
        for i, (x, y) in enumerate(test_data):
            exs = test_exs[i]
            outputs = self.forward(x)
            _, predicted = torch.max(torch.round(torch.sigmoid(outputs)), 1)
            predictions.append(SentimentExample(exs.indexed_words, predicted[0].item()))
        return predictions


# , using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value).
def train_evaluate_ffnn(train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                        test_exs: List[SentimentExample],
                        word_vectors: WordEmbeddings,
                        number_layer=1) -> List[SentimentExample]:
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


    num_epochs = 20
    num_data = len(train_data)
    ffnn = FFNN(embedding_size, hidden_size, num_classes, num_layer=number_layer)
    #optimizer = torch.optim.Adam(ffnn.parameters(), lr=0.1)
    #optimizer = torch.optim.SGD(ffnn.parameters(), lr=0.1)
    #optimizer = torch.optim.Adadelta(ffnn.parameters())
    optimizer = torch.optim.Adagrad(ffnn.parameters())

    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, num_epochs):
        start_time = time.time()
        ex_indices = [i for i in range(num_data)]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x, y = train_data[idx]
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            optimizer.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            #loss = torch.neg(torch.log(probs)).dot(y)
            loss = criterion(probs, y)
            total_loss += loss
            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # evaluate for each epoch
        train_loss, train_acc = ffnn.evaluate(train_data, criterion)
        dev_loss, dev_acc = ffnn.evaluate(dev_data, criterion)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')

    prediction = ffnn.predict(test_exs, word_vectors)
    return prediction



