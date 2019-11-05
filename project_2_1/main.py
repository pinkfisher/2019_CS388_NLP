import argparse
import time
import math
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List
from masked_cross_entropy import *

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true', help='set to debug mode ')
    parser.add_argument('--num_train_sentence', dest='num_train_sentence', type=int, default=-1,
                        help='set number of sentence to train on')
    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    args = parser.parse_args()
    return args


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self, encoder, decoder, input_indexer, output_indexer, input_max_len, output_max_len, attention=False):
        self.encoder = encoder
        self.decoder = decoder
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.attention = attention
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len

    def decode(self, all_test_data: List[Example]) -> List[List[Derivation]]:
        test_derivs = []
        for test_data in all_test_data:
            test_data = [test_data]

            # Create indexed input
            input_lengths = np.asarray([len(ex.x_indexed) for ex in test_data])
            input_max_len = np.max(input_lengths)
            input_batches = make_padded_input_tensor(test_data, input_indexer, input_max_len, reverse_input=False)
            input_batches = torch.from_numpy(input_batches).transpose_(0, 1)

            decoded_words, _ = evaluate_batch(input_batches, input_lengths, self.output_max_len,
                                              self.encoder, self.decoder, self.output_indexer)
            test_derivs.append([Derivation(test_data, 1.0, decoded_words)])

        return test_derivs

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def random_batch(all_train_data: List[Example], batch_size: int, input_max_len, output_max_len, input_indexer, output_indexer):
    train_data = list(np.random.choice(all_train_data, batch_size, replace=False))

    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_lengths = np.asarray([len(ex.x_indexed) for ex in train_data])

    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_train_input_data = torch.from_numpy(all_train_input_data).transpose_(0, 1)

    target_lengths = np.asarray([len(ex.y_indexed) for ex in train_data])

    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_train_output_data = torch.from_numpy(all_train_output_data).transpose_(0, 1)

    return all_train_input_data, input_lengths, all_train_output_data, target_lengths


def train_model_encdec(all_train_data: List[Example], test_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param test_data:
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    print("==========START DATA PROCESSING=========")
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in all_train_data]))
    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in all_train_data]))
    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % output_max_len)

    if args.debug:
        output_indexer = input_indexer

    # Configuration
    attn_model = "general"
    hidden_size = 100
    n_layers = 2
    dropout = 0.1
    teacher_forcing_ratio = 0.5
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0

    # Initialize models
    encoder = EncoderRNN(len(input_indexer), hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, len(output_indexer), n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    print("==========START TRAINING=========")
    start = time.time()
    print_loss_total = 0
    for epoch in range(1, args.epochs+1):
        for iter in range(int(len(all_train_data)/args.batch_size)):

            input_batches, input_lengths, target_batches, target_lengths = random_batch(all_train_data, args.batch_size,
                                                            input_max_len, output_max_len, input_indexer, output_indexer)

            if args.debug:
                target_batches = input_batches
                target_lengths = input_lengths

            # Run the train function
            loss, ec, dc = train_batch(
                input_batches, input_lengths, target_batches, target_lengths,
                encoder, decoder, encoder_optimizer, decoder_optimizer, output_max_len,
                criterion, args
            )
            print_loss_total += loss
        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / args.epochs),
                                     epoch, epoch / args.epochs * 100, print_loss_total))
        print_loss_total = 0

    return Seq2SeqSemanticParser(encoder, decoder, input_indexer, output_indexer, input_max_len, output_max_len)


def train_batch(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, output_max_len, criterion, args):

    SOS_token = output_indexer.index_of("<SOS>")
    EOS_token = output_indexer.index_of("<EOS>")

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * args.batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(output_max_len, args.batch_size, decoder.output_size))


    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    #loss = criterion(all_decoder_outputs.transpose(0, 1).contiguous(), target_batches.transpose(0, 1).contiguous())
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), 50.0)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), 50.0)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data, ec, dc


def evaluate_batch(input_batches, input_lengths, max_length, encoder, decoder, output_indexer):

    SOS_token = output_indexer.index_of("<SOS>")
    EOS_token = output_indexer.index_of("<EOS>")

    with torch.no_grad():

        # Run through encoder
        #print(input_batches.shape)
        #print(input_lengths)
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([SOS_token]))  # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                #decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_indexer.get_object(ni.item()))

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    java_crashes = False
    if java_crashes:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    else:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)

    example_freq = 50
    if 0 < args.num_train_sentence < example_freq:
        example_freq = 1

    print("=======TRAIN EVALUATION=======")
    evaluate(train_data_indexed, decoder, example_freq=example_freq)
    print("=======END OF TRAIN EVALUATION=======")
    print("=======DEV EVALUATION=======")
    evaluate(dev_data_indexed, decoder, example_freq=25)
    print("=======END OF DEV EVALUATION=======")

    if not args.debug:
        print("=======FINAL EVALUATION ON BLIND TEST=======")


