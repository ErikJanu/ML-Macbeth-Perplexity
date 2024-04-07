from nltk.tokenize import word_tokenize
import math
import re
import collections


## nltk.download('punkt')

class Corpus:
    def __init__(self, file_path):
        self.file_path = file_path
        self.contents = None
        self.train_corpus = None
        self.test_corpus = None

    def read_file(self):
        with open(self.file_path, "r") as file:
            self.contents = file.read()

    def clean_file(self):
        self.contents = re.sub(r"[^\w\s]", "", self.contents.lower())
        self.contents = re.sub(r"[\d]", "", self.contents)

    def file_tokenize(self):
        self.contents = word_tokenize(self.contents)

    def split_file(self):
        slice_helper = int(len(self.contents) * 0.8)
        self.train_corpus = self.contents[:slice_helper]
        self.test_corpus = self.contents[slice_helper:]


class Vocabulary:
    def __init__(self, text):
        self.vocabulary = sorted(set(text))

    def write_vocabulary(self, file_path):
        with open(file_path, "w") as vocab:
            vocab.write(" ".join(self.vocabulary))


class Ngrams:
    def __init__(self, text):
        self.text = text
        self.unigrams_train = None
        self.unigrams_train_freq = None
        self.unigrams_probs = None
        self.bigrams_train = None
        self.bigrams_train_freq = None
        self.bigrams_probs = None

    def generate_ngrams(self):
        self.unigrams_train = [word for word in self.text]
        self.bigrams_train = [(self.text[i], self.text[i + 1]) for i in range(len(self.text) - 1)]

    def count_ngrams(self):
        self.bigrams_train_freq = collections.Counter(self.bigrams_train)
        self.unigrams_train_freq = collections.Counter(self.unigrams_train)

    def write_bigrams_to_file(self, file_path):
        ngram_dict = {key: value for key, value in self.bigrams_train_freq.items()}
        with open(file_path, "w") as ngram_file:
            ngram_file.write(str(ngram_dict))

    def calculate_probabilities(self):
        num_of_bigrams = len(self.bigrams_train_freq)
        num_of_unigrams = len(self.unigrams_train_freq)
        self.bigrams_probs = {ngram: float(frequency / num_of_bigrams) for ngram, frequency in
                              self.bigrams_train_freq.items()}
        self.unigrams_probs = {ngram: float(frequency / num_of_unigrams) for ngram, frequency in
                               self.unigrams_train_freq.items()}


class Perplexity:
    def __init__(self, ngrams, test_corpus):
        self.test_corpus = test_corpus
        self.train_bigram_freq = ngrams.bigrams_train_freq
        self.train_bigram_prob = ngrams.bigrams_probs
        self.train_unigram_freq = ngrams.unigrams_train_freq

    def calculate_perplexity(self, n):
        total_words_test = len(self.test_corpus)
        perplexity_sum = 0

        for i in range(len(self.test_corpus) - n + 1):
            # Extract the n-gram and its context
            ngram = tuple(self.test_corpus[i:i + n])
            context = ngram[:-1]

            # calculate perplexity using formula for perplexity (2 ^ -(1/N)  * sum log 2 probabilty of each word (in
            # test set) given its probability (by frequency [frequency bigram divided by frequency unigram) in the training set
            ngram_probability = self.train_bigram_freq.get(ngram, 0) / self.train_unigram_freq.get(context, 1)

            if ngram_probability > 0:
                perplexity_sum += math.log2(ngram_probability)

        perplexity = 2 ** (-perplexity_sum / total_words_test)
        return perplexity


corpus = Corpus("shakespeare-macbeth.txt")
corpus.read_file()
corpus.clean_file()
corpus.file_tokenize()
corpus.split_file()

vocabulary_train = Vocabulary(corpus.train_corpus)
vocabulary_test = Vocabulary(corpus.test_corpus)

vocabulary_train.write_vocabulary("vocabulary_train.txt")
vocabulary_test.write_vocabulary("vocabulary_test.txt")

ngrams_train = Ngrams(corpus.train_corpus)
ngrams_train.generate_ngrams()
ngrams_train.count_ngrams()
ngrams_train.write_bigrams_to_file("train_bigrams.txt")
ngrams_train.calculate_probabilities()

print("Probabilites of bigrams in the training corpus: ", ngrams_train.bigrams_probs)

perplexity_calc = Perplexity(ngrams_train, corpus.test_corpus)
perplexity_score = perplexity_calc.calculate_perplexity(2)
print("Perplexity for bigrams in the test corpus: ", perplexity_score)
