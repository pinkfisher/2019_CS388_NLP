# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np
import time
import os


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode1(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs,
                                             self.transition_log_probs, self.emission_log_probs)
        return viterbi(sentence_tokens, scorer)


    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self, sentence):
        pred_tags = []
        scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs,
                                             self.transition_log_probs, self.emission_log_probs)

        # Implements the viterbi algorithm based upon fig 10.8 of the book "Speech and Language Processing (3rd ed. draft)"
        # NOTE: since the whole code adopts the $\pi, QA$ representation (see Section 9.2), we don't need the start and end state.
        T = len(sentence)  # Number of observations
        N = len(self.tag_indexer)  # Number of states
        viterbi = np.zeros(shape=(N, T))  # Create a path probability matrix viterbi[N,T]
        backpointer = np.zeros(shape=(N, T))

        # Initialization step
        for s in range(N):
            # "+" because the probabilities are log-based
            viterbi[s, 0] = scorer.score_init(sentence, s) + scorer.score_emission(sentence, s, 0)
            backpointer[s, 0] = 0

        # Recursion step
        for t in range(1, T):
            for s in range(N):
                tmp1 = np.zeros(N)  # build the candidate values for viterbi
                tmp2 = np.zeros(N)  # build the candidate values for backpointer
                for s_tmp in range(N):
                    # "+" because the probabilities are log-based
                    tmp1[s_tmp] = viterbi[s_tmp, t - 1] + scorer.score_transition(sentence, s_tmp,
                                                                                  s) + scorer.score_emission(sentence,
                                                                                                             s, t)
                    tmp2[s_tmp] = viterbi[s_tmp, t - 1] + scorer.score_transition(sentence, s_tmp, s)
                viterbi[s, t] = np.max(tmp2) + scorer.score_emission(sentence, s, t)
                backpointer[s, t] = np.argmax(tmp2)

        # Termination step (skipped because we don't have the end state)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(viterbi[:, T - 1])))
        for t in range(1, T):
            pred_tags.append(self.tag_indexer.get_object(backpointer[self.tag_indexer.index_of(pred_tags[-1]), T - t]))

        pred_tags = list(reversed(pred_tags))
        return LabeledSentence(sentence, chunks_from_bio_tag_seq(pred_tags))


def viterbi(sentence: List[Token], scorer: ProbabilisticSequenceScorer):
    N = len(sentence)  # num_tokens
    T = len(scorer.tag_indexer)  # num_tags
    v = np.zeros((N, T))  # prob for observing x = v_i when hidden state y = v_j
    y_max_prev = np.zeros((N, T))    # which y_prev gives the best y_current

    # Initial states
    for y in range(T):  # for all possible state y
        # prob of having state y, producing the first token
        v[0, y] = scorer.score_init(sentence, y) + scorer.score_emission(sentence, y, 0)

    for i in range(1, N):   # for every other token in the sentence
        for y in range(T):      # for every state y
            previous_prob = np.zeros(T)
            for y_prev in range(T):
                previous_prob[y_prev] = scorer.score_transition(sentence, y_prev, y) + v[i-1, y_prev]
            v[i, y] = scorer.score_emission(sentence, y, i) + np.max(previous_prob)
            y_max_prev[i, y] = np.argmax(previous_prob)

    idx = np.argmax(v[-1, :])   # find the most possible last state
    pred_tags = [(scorer.tag_indexer.get_object(idx))]
    for t in range(1, N):  # trace back
        idx = y_max_prev[N - t, idx]
        pred_tags.append(scorer.tag_indexer.get_object(idx))

    pred_tags.reverse()

    return LabeledSentence(sentence, chunks_from_bio_tag_seq(pred_tags))


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence):
        # Start a timer
        # start_time = time.time()

        pred_tags = []
        T = len(sentence)
        N = len(self.tag_indexer)
        viterbi = np.zeros(shape=(N, T))
        backpointer = np.zeros(shape=(N, T))

        score_matrix = np.zeros(shape=(N, T))
        for s in range(N):
            for t in range(T):
                features = extract_emission_features(sentence,
                                                     t,
                                                     self.tag_indexer.get_object(s),
                                                     self.feature_indexer,
                                                     add_to_indexer=False)
                score = sum([self.feature_weights[i] for i in features])
                score_matrix[s, t] = score

        # Initialization step
        for s in range(N):
            # "+" because the probabilities are log-based
            tag = str(self.tag_indexer.get_object(s))
            if (isI(tag)):
                viterbi[s, 0] = float("-inf")
            else:
                viterbi[s, 0] = score_matrix[s, 0]
            backpointer[s, 0] = 0

        # Recursion step
        for t in range(1, T):
            for s in range(N):
                tmp1 = np.zeros(N)
                tmp2 = np.zeros(N)
                for s_tmp in range(N):
                    # "+" because the probabilities are log-based
                    # We want to ban out certain scenario:
                    # 1. We cannot have O, I tag sequence of any type
                    # 2. We cannot have I-x, I-y tag sequence of different types
                    # 3. We cannot have B-x, I-y tag sequence of any type of I other than x
                    prev_tag = str(self.tag_indexer.get_object(s_tmp))
                    curr_tag = str(self.tag_indexer.get_object(s))
                    if (isO(prev_tag) and isI(curr_tag)) or \
                            (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) or \
                            (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
                        tmp1[s_tmp] = float("-inf")
                        tmp2[s_tmp] = float("-inf")
                    else:
                        tmp1[s_tmp] = viterbi[s_tmp, t - 1] + score_matrix[s, t]
                        tmp2[s_tmp] = viterbi[s_tmp, t - 1]
                viterbi[s, t] = np.max(tmp1)
                backpointer[s, t] = np.argmax(tmp2)
                # Termination step (skipped because we don't have the end state)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(viterbi[:, T - 1])))
        for t in range(1, T):
            pred_tags.append(self.tag_indexer.get_object(backpointer[self.tag_indexer.index_of(pred_tags[-1]), T - t]))

        pred_tags = list(reversed(pred_tags))

        # Calculate the amount of time used for one sentence
        # The actual time per this function call is around 1s
        # elapsed_time = time.time() - start_timeß
        # hours, rem = divmod(elapsed_time, 3600)
        # minutes, seconds = divmod(rem, 60)
        # print('[viterbi] time eplased: {:0>2}:{:05.2f}'.format(int(minutes), seconds))

        return LabeledSentence(sentence, chunks_from_bio_tag_seq(pred_tags))


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    print("Training")

    num_sentences = int(len(sentences))
    feature_weights = np.random.rand(len(feature_indexer))

    optimizer = SGDOptimizer(feature_weights, 0.1)

    total_num_epoch = 5
    for epoch in range(total_num_epoch):
        loss = 0
        # Start a timer
        start_time = time.time()

        # For each sentence in the training set
        for sentence_idx in range(num_sentences):
            gradients = Counter()

            T = len(sentences[sentence_idx])  # Number of observations
            N = len(tag_indexer)  # Number of states

            # Construct feature matrix
            feature_matrix = np.zeros(shape=(N, T))
            for s in range(N):
                for t in range(T):
                    # Calculate $\phi_e(y_i,i,\pmb{x})$
                    feature_matrix[s, t] = np.sum(np.take(feature_weights, feature_cache[sentence_idx][t][s]))

            forward = np.zeros(shape=(N, T))  # create a matrix to store the forward probabilities
            backward = np.zeros(shape=(N, T))  # create a matrix to store the backward probabilities

            #   Forward-backward algorithm to calculate P(y_i = s | X)
            #   NOTE: I ignore transition feature for now

            #  Forward
            # Initialization step
            for s in range(N):
                forward[s, 0] = feature_matrix[s, 0]

            # Recursion step
            for t in range(1, T):
                for s in range(N):
                    # sum = logsumexp(forward[:,t-1])
                    sum = 0
                    for i in range(N):
                        if i == 0:
                            sum = forward[i, t - 1]
                        else:
                            sum = np.logaddexp(sum, forward[i, t - 1])
                    forward[s, t] = feature_matrix[s, t] + sum

            # Backward
            # Initialization step
            for s in range(N):
                backward[s, T - 1] = 0  # alternatively, backward[:,-1] = 0

            # Recursion step
            for t in range(1, T):
                for s in range(N):
                    sum = 0
                    for i in range(N):
                        if i == 0:
                            sum = backward[i, T - t] + feature_matrix[i, T - t]
                        else:
                            sum = np.logaddexp(sum, backward[i, T - t] + feature_matrix[i, T - t])
                    backward[s, T - t - 1] = sum

            # Calculate normalizing constant Z in log space
            # Z is a constant. Since the last column of the backward matrix contains all 0s,
            # we can use the last column to avoid using backward matrix.
            Z = 0
            for s in range(N):
                if s == 0:
                    Z = forward[s, -1]
                else:
                    Z = np.logaddexp(Z, forward[s, -1])

            # Check if Z value if the same
            test_normalizing_constant = False
            if test_normalizing_constant:
                Z1 = forward[0, 0]+backward[0, 0]
                for s in range(1, N):
                    Z1 = np.logaddexp(Z1, forward[s, 0]+backward[s, 0])
                np.testing.assert_almost_equal(Z, Z1)

            # Compute the posterior probability -P(y_i = s | X)
            p_y_s_x = np.zeros(shape=(N, T))
            for s in range(N):
                for t in range(T):
                    p_y_s_x[s, t] = np.exp(forward[s, t] + backward[s, t] - Z)

            #  Compute the stochastic gradient of the feature vector for a sentence
            #  gradients = sum of gold features - expected features under model
            for word_idx in range(len(sentences[sentence_idx])):
                # Find the gold tag for the given word
                gold_tag = tag_indexer.index_of(sentences[sentence_idx].get_bio_tags()[word_idx])
                features = feature_cache[sentence_idx][word_idx][gold_tag]
                loss += np.sum([feature_weights[i] for i in features])
                for feature in features:
                    gradients[feature] += 1  # feature value is 0 or 1

                # Calculate expected features = p(y_i = s | x) * feature
                for tag_idx in range(N):
                    features = feature_cache[sentence_idx][word_idx][tag_idx]
                    for feature in features:
                        gradients[feature] -= p_y_s_x[tag_idx, word_idx]

            # Update the weights using the gradient computed
            loss -= Z
            optimizer.apply_gradient_update(gradients, 10)

        # Calculate the amount of time used for one epoch
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('epoch: {} time eplased: {:0>2}:{:05.2f}. loss: {}'.format(epoch, int(minutes), seconds, loss))

        # Run bunch of experimentations to gather the data for plot
        if os.getenv('CRF_EXP', default=False):
            crf_model = CrfNerModel(tag_indexer, feature_indexer, optimizer.get_final_weights())
            if os.getenv('CRF_ENG', default=False):
                # Gather the data on Accuracy vs. epoch on dev set
                dev = read_data("data/eng.testa")
            elif os.getenv('CRF_DEU', default=False):
                dev = read_data("data/deu.testa")
            dev_decoded = [crf_model.decode(test_ex) for test_ex in dev]
            print_evaluation(dev, dev_decoded)
    return CrfNerModel(tag_indexer, feature_indexer, optimizer.get_final_weights())


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)

    # Character n-grams of the current word
    """
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    """
    return np.asarray(feats, dtype=int)

