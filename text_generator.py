#-*- coding: utf-8 -*-
import re
import argparse
import sys
import codecs
import numpy
import unittest
from collections import Counter
from collections import defaultdict
from itertools import islice
from itertools import izip


def tokenize(line):
    letters_seq = ur'[A-Za-zА-Яа-я]+'
    digits_seq = ur'[0-9]+'
    other = ur'[^A-Za-zА-Яа-я0-9]'
    pattern = letters_seq + ur'|' + digits_seq + ur'|' + other
    tokens = re.findall(pattern, line, re.U)
    return tokens


def print_tokens(tokens):
    for token in tokens:
        print token


def run_tokenize(line):
    print_tokens(tokenize(line))


def extract_words(line):
    return re.findall(ur'[A-Za-zА-Яа-я]+', line, re.U)


def build_slices(current_depth, words):
    slices = []
    for shift in xrange(current_depth + 1):
        slices.append(islice(words, shift, None))
    return slices


def make_frequencies(counter):
    values_sum = sum(counter.values())
    for key in counter:
        counter[key] = float(counter[key]) / values_sum


def find_probabilities(lines, depth):
    counters = defaultdict(Counter)
    for line in lines:
        words = extract_words(line)
        for current_depth in xrange(depth + 1):
            for words_chain in izip(*build_slices(current_depth, words)):
                counters[words_chain[:-1]][words_chain[-1]] += 1
    for key in counters:
        make_frequencies(counters[key])
    return counters


def print_probabilities(counters):
    sorted_counters = sorted(counters.iteritems(), key=lambda x: x[0])
    for chain, counter in sorted_counters:
        print ur' '.join(chain)
        sorted_counter = sorted(counter.iteritems(), key=lambda x: x[0])
        for word, proba in sorted_counter:
            print ur"  {0}: {1:.2f}".format(word, proba)


def run_probabilites(lines, depth):
    print_probabilities(find_probabilities(lines, depth))


def generate_word(freqs):
    return numpy.random.choice(freqs.keys(), 1, p=freqs.values())[0]


def update_history(current_history, current_word, depth):
    if len(current_history) == depth:
        current_history[:] = current_history[1:]
    current_history.append(current_word)


def next_word(current_history, probabilities, depth):
    prob_dist = probabilities[tuple(current_history)]
    if len(prob_dist) == 0:
        current_history[:] = []
        prob_dist = probabilities[tuple(current_history)]
    current_word = generate_word(prob_dist)
    update_history(current_history, current_word, depth)
    return current_word


def generate_text(lines, depth, words_count, dot_probability=0.2):
    lines = [line.lower() for line in lines]
    probabilities = find_probabilities(lines, depth)
    current_history = []
    text = []
    sentence = []
    set_dot = False
    for i in xrange(words_count):
        if len(sentence) > 3:
            set_dot = numpy.random.choice([True, False], 1, p=(dot_probability,
                                                               1.0-dot_probability))[0]
        if set_dot:
            sentence[0] = sentence[0].title()
            sentence[-1] += '.'
            text += sentence
            current_history = []
            sentence = []
            set_dot = False
        else:
            sentence.append(next_word(current_history, probabilities, depth))
    return ' '.join(text)


class Test(unittest.TestCase):
    def test_probabilities(self):
        test_line = "Hello, world!"
        extracted_words = extract_words(test_line)
        words = ['Hello', 'world']
        self.assertEqual(extracted_words, words)

    def test_tokenize(self):
        test_line = "Hello, world!"
        tokenized = ['Hello', ',', ' ', 'world', '!']
        self.assertEqual(tokenize(test_line),tokenized)

    def test_generate(self):
        training = ['Life, it seems, will fade away',
                    'Drifting further every day',
                    'Getting lost within myself',
                    'Nothing matters, no one else']
        text = generate_text(lines=training, depth=3,
                             words_count=100, dot_probability=0.2)
        self.assertGreater(len(text), 0)


def run_tests():
    unittest.main()


def build_parser():
    modes = ('tokenize', 'probabilities', 'generate', 'test')
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', metavar='mode', type=str, nargs=1,
                        help='working mode', choices=modes)
    parser.add_argument('--depth', metavar='depth', type=int, nargs=1,
                        help='depth')
    parser.add_argument('--size', metavar='size', type=int, nargs=1,
                        help='depth')
    return parser


def run(args):
    if args.mode[0] != 'test':
        raw_input_lines = sys.stdin.readlines()
        input_lines = [line[:-1] for line in raw_input_lines]
        if args.mode[0] == 'tokenize':
            run_tokenize(input_lines[0])
        elif args.mode[0] == 'probabilities':
            run_probabilites(input_lines, args.depth[0])
        elif args.mode[0] == 'generate':
            generate_text(input_lines, args.depth[0], args.size[0])
    else:
        run_tests()


if __name__ == '__main__':
    raw_args = sys.stdin.readline().split()
    args = build_parser().parse_args(raw_args)

    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    run(args)
