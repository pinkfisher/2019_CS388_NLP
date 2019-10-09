from sentiment_data import *
from typing import List
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

"""
    CNN Model for sentiment analysis
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
    train_mat = [pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs]

    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # Select a subset of data
    if num_data:
        train_mat = train_mat[:num_data, :]
        train_labels_arr = train_labels_arr[:num_data]

    train_data = []
    for idx in range(len(train_mat)):
        x = form_input(train_mat[idx], word_vectors, average=False)
        y = train_labels_arr[idx]
        y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        train_data.append((x, y))
    print("number of 1s: {}".format(np.sum(train_labels_arr)))
    return train_data


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class CNNModel(nn.Module):
    def __init__(self, embedding_size, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_size))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_size))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_size))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        #text = text.permute(1, 0)

        # text = [batch size, sent len]

        #embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        text = text.float()
        embedded = text.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        activation = "ReLu"

        if activation == "ReLu":
            f = nn.ReLU()
        elif activation == "Tanh":
            f = nn.Tanh()

        conved_0 = f(self.conv_0(embedded).squeeze(3))
        conved_1 = f(self.conv_1(embedded).squeeze(3))
        conved_2 = f(self.conv_2(embedded).squeeze(3))

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def get_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        _, rounded_preds = torch.max(torch.sigmoid(preds), 1)
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def evaluate(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        # Sets the module in evaluation mode
        self.eval()

        # Disable gradient calculation
        with torch.no_grad():
            for (x, y) in iterator:
                predictions = self(x).squeeze(1)

                loss = criterion(predictions, y)

                acc = self.get_accuracy(predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def predict(self, test_exs: List[SentimentExample], word_vectors):
        test_data = prepare_dataset(test_exs, word_vectors)
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
def train_evaluate_CNN(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], test_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> List[SentimentExample]:
    embedding_size = word_vectors.get_embedding_length()
    num_classes = 2
    batch_size = 32
    learning_rate = 0.1
    n_filters = 100
    #n_filters = 200
    #n_filters = 300
    #filter_size = [3, 4, 5]
    filter_size = [2, 3, 4]
    #filter_size = [4, 5, 6]
    #filter_size = [3, 3, 3]
    dropout = 0.5

    train_data = prepare_dataset(train_exs, word_vectors, num_data=0)
    dev_data = prepare_dataset(dev_exs, word_vectors)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dataset=dev_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    num_epochs = 20
    model = CNNModel(embedding_size, n_filters, filter_size, num_classes, dropout)
    model = model.float()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adadelta(model.parameters())
    # optimizer = torch.optim.Adagrad(model.parameters())

    criterion = nn.CrossEntropyLoss()

    best_dev_acc = float('-inf')
    for epoch in range(num_epochs):
        start_time = time.time()
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

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # evaluate for each epoch
        train_loss, train_acc = model.evaluate(train_loader, criterion)
        dev_loss, dev_acc = model.evaluate(dev_loader, criterion)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            test_exs_predicted = model.predict(test_exs, word_vectors)

            write_sentiment_examples(test_exs_predicted, 'test-blind.output.txt', word_vectors.word_indexer)
            print("======== Finish writing output ========")

    return test_exs_predicted

