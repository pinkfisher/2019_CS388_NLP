# sentiment.py

import argparse
from models import *
from model_LSTM import *
from model_CNN import *
from sentiment_data import *

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='CNN', help='model to run (FF or FANCY or CNN)')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.50d-relativized.txt', help='path to word vectors file')
    parser.add_argument('--word_vecs_dim', type=int, default=300, help='dimention of word vector embeddings')
    parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings(args.word_vecs_path)
    if args.word_vecs_dim == 50:
        word_vectors = read_word_embeddings('data/glove.6B.50d-relativized.txt')
    elif args.word_vecs_dim == 300:
        word_vectors = read_word_embeddings('data/glove.6B.300d-relativized.txt')

    # Load train, dev, and test exs
    train_exs = read_and_index_sentiment_examples(args.train_path, word_vectors.word_indexer)
    dev_exs = read_and_index_sentiment_examples(args.dev_path, word_vectors.word_indexer)
    test_exs = read_and_index_sentiment_examples(args.blind_test_path, word_vectors.word_indexer)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples")

    if args.model == "FF":
        test_exs_predicted = train_evaluate_ffnn(train_exs, dev_exs, test_exs, word_vectors, number_layer=1)
        write_sentiment_examples(test_exs_predicted, args.test_output_path, word_vectors.word_indexer)
    elif args.model == "FANCY":
        test_exs_predicted = train_evaluate_fancy(train_exs, dev_exs, test_exs, word_vectors)
    elif args.model == "CNN":
        test_exs_predicted = train_evaluate_CNN(train_exs, dev_exs, test_exs, word_vectors)
    else:
        raise Exception("Pass in either FF or FANCY to run the appropriate system")