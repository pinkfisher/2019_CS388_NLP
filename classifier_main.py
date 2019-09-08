# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
import numpy as np

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    #parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--train_path', type=str, default='data/eng_small.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


class PersonExample(object):
    """
    Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

    Attributes:
        tokens: the sentence to classify
        labels: 0 if non-person name, 1 if person name for each token in the sentence
    """
    def __init__(self, tokens: List[str], labels: List[int]):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        rst = ""
        for i in range(len(self.tokens)):
            rst += self.tokens[i] + ": " + str(self.labels[i]) + "\n"
        return rst


def transform_for_classification(ner_exs: List[LabeledSentence]):
    """
    :param ner_exs: List of chunk-style NER examples
    :return: A list of PersonExamples extracted from the NER data
    """
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels)


class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[str], idx: int):
        if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
            return 1
        else:
            return 0


def train_count_based_binary_classifier(ner_exs: List[PersonExample]):
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts[ex.tokens[idx]] += 1.0
            else:
                neg_counts[ex.tokens[idx]] += 1.0
    print(repr(pos_counts))
    print(repr(pos_counts["Peter"]))
    print(repr(pos_counts["aslkdjtalk;sdjtakl"]))
    return CountBasedPersonClassifier(pos_counts, neg_counts)


class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer
        self.optimizer = SGDOptimizer(weights, 0.01)

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
    def predict(self, tokens, idx):
        feature = get_feature(tokens, idx, self.indexer)
        if sigmoid(np.dot(self.weights, feature)) > 0.5:
            return 1
        else:
            return 0

    def train(self, ner_exs: List[PersonExample]):
        # gradient = xj (yj - logistic(w * xj))

        # def apply_gradient_update(self, gradient: Counter, batch_size: int):
        #   for i in gradient.keys():
        #       self.weights[i] = self.weights[i] + self.alpha * gradient[i]
        for k, ex in enumerate(ner_exs):
            if k/100 == 0:
                print("Update example " + str(k))
                print("Finished " + str(k/len(ner_exs)) + "%")

            for idx in range(0, len(ex)):
                label = ex.labels[idx]

                # generate feature
                # one-hot representation of each word
                feature = get_feature(ex.tokens, idx, self.indexer)

                # calculate gradient update
                gradient = np.dot(feature, label - sigmoid(np.dot(self.weights, feature)))
                gradients = array_to_counter(gradient)
                self.optimizer.apply_gradient_update(gradients, 10)
                self.weights = self.optimizer.get_final_weights()

        print(self.weights)


def get_feature(tokens: List[str], idx: int, indexer: Indexer):
    feature = np.zeros(len(indexer.objs_to_ints))
    for token in tokens:
        np.put(feature, indexer.index_of(token), 1)
    return feature


def write_features(ner_exs: List[PersonExample], indexer, out_path="data/features"):
    with open(out_path, "w") as f:
        for k, ex in enumerate(ner_exs):
            for idx in range(0, len(ex)):
                feature = get_feature(ex.tokens, idx, indexer)
                feature_str = " ".join(str(x) for x in feature)
                feature_str += "\n"
                f.write(feature_str)

def array_to_counter(input: np.array):
    output = Counter()
    idx = 0
    for x in np.nditer(input):
        output[idx] = x
        idx += 1
    return output


def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))


def train_classifier(ner_exs: List[PersonExample]):
    # build vocabulary
    indexer = Indexer()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            token = ex.tokens[idx]
            indexer.add_and_get_index(token)

    initial_weights = np.zeros(len(indexer.objs_to_ints))

    classifier = PersonClassifier(initial_weights, indexer)

    #write_features(train_class_exs, indexer)

    classifier.train(ner_exs)

    return classifier


def evaluate_classifier(exs: List[PersonExample], classifier: PersonClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: PersonExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            predictions.append(classifier.predict(ex.tokens, idx))
    print_evaluation(golds, predictions)


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


def predict_write_output_to_file(exs: List[PersonExample], classifier: PersonClassifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))

    for i in dev_class_exs:
        print(i)

    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs)
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



