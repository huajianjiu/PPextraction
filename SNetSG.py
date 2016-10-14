from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.models.embedding.word2vec import Options, Word2Vec
from WordnetNet import WordnetNet

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np

from tensorflow.models.embedding import gen_word2vec as word2vec


flags = tf.app.flags
# flags.DEFINE_integer("relations_num", 5,
#                      "The number of related words to be considered")
FLAGS = flags.FLAGS

bug_f = open("debug.txt", "a+")

class OptionsSsg(Options):
    def __init__(self):
        super(OptionsSsg, self).__init__()
        self.relations_num = FLAGS.relations_num


class SNetSgAverage(Word2Vec):
    def __init__(self, options, session, syn_relations):
        self._syn_relations = tf.constant(syn_relations)
        super(SNetSgAverage, self).__init__(options, session)

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits, tf.ones_like(true_logits))
        # true_xent = tf.check_numerics(true_xent, "true_xent nan")

        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits, tf.zeros_like(sampled_logits))
        # sampled_xent = tf.check_numerics(sampled_xent, "sampled_xent nan")

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def forward(self, examples, labels):

        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.

        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")
        self._emb = emb

        # Word - related word: [vocab_size, relations_num]
        wrw = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.relations_num], -init_width, init_width),
            name="wrw")
        self._wrw = wrw

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # ids for synonyms of the examples: [batch_size, relations_num]
        synonyms = tf.nn.embedding_lookup(self._syn_relations, examples)
        # flattern synonyms so we can get the embeddings of them
        synonyms_flat = tf.reshape(synonyms, [-1])
        # synonyms_emb: [batch_size*relations_num, emb_dim]
        synonyms_emb = tf.nn.embedding_lookup(emb, synonyms_flat)

        # get weighted average of the embedding of synonyms
        # TODO: this is one of the ideas. remember to try:
        # TODO: 1.use trusted  paraphrase database
        # TODO: 2.use a softmax with biased threshold to train the synonym-layer to
        # TODO: trusted  paraphrase database
        # change synonyms_emb to synonyms_emb_reshape:[batch_size, relations_num, emb_dim]
        synonyms_emb_reshaped = tf.reshape(synonyms_emb,
                                          [opts.batch_size, opts.relations_num, opts.emb_dim])
        synonyms_emb_reshaped = tf.check_numerics(synonyms_emb_reshaped, "synonyms_emb_reshaped")
        # weight the synonyms embeddings [batch_size, relations_num, emb_dim]
        wrw_weights = tf.nn.embedding_lookup(wrw, examples)
        wrw_weights = tf.expand_dims(wrw_weights, 2)
        synonyms_emb_reshaped_weighted = tf.mul(synonyms_emb_reshaped, wrw_weights)

        # the 1st method getting the embedding
        synonyms_emb_reshaped_weighted_mean = tf.reduce_mean(synonyms_emb_reshaped_weighted, 1)

        synonyms_emb_reshaped_weighted_mean = tf.check_numerics(synonyms_emb_reshaped_weighted_mean,
                                                                  "syn_emb_reshaped_weighted_mean nan or inf")

        example_emb_syn_mean = tf.reduce_mean(
            tf.reshape(
                tf.concat(1, [example_emb, synonyms_emb_reshaped_weighted_mean]),
                [opts.batch_size, 2, opts.emb_dim]
            ), 1
        )
        example_emb_syn_mean = tf.check_numerics(example_emb_syn_mean, "emb_syn_mean nan or inf")

        # the 2nd method getting the mean embedding of the synonyms
        # sometimes diverges.  do not know why.
        # example_emb_syn = tf.concat(1,
        #                             [tf.expand_dims(example_emb, 1),
        #                              synonyms_emb_reshaped_weighted])
        # example_emb_syn_mean = tf.reduce_mean(example_emb_syn, 1)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(example_emb_syn_mean, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb_syn_mean,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        true_logits = tf.check_numerics(true_logits, "true_logits nan")
        sampled_logits = tf.check_numerics(sampled_logits, "sampled_logits nan")
        return true_logits, sampled_logits

    def get_final_emb(self):
        # get the averaged embeddings of the input and its related words as the output embeddings
        opts = self._options
        synonyms_flat = tf.reshape(self._syn_relations, [-1])
        synonyms_emb = tf.nn.embedding_lookup(self._emb, synonyms_flat)
        synonyms_emb_reshaped = tf.reshape(synonyms_emb,
                                           [opts.vocab_size, opts.relations_num, opts.emb_dim])
        wrw_weights = tf.expand_dims(self._wrw, 2)
        synonyms_emb_reshaped_weighted = tf.mul(synonyms_emb_reshaped, wrw_weights)
        synonyms_emb_reshaped_weighted_mean = tf.reduce_mean(synonyms_emb_reshaped_weighted, 1)
        emb_syn_mean = tf.reduce_mean(
            tf.reshape(
                tf.concat(1, [self._emb, synonyms_emb_reshaped_weighted_mean]),
                [opts.vocab_size, 2, opts.emb_dim]
            ), 1
        )
        return emb_syn_mean


    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        emb = self.get_final_emb()
        nemb = tf.nn.l2_normalize(emb, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, self._options.vocab_size))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx


class SNetSgAvReNoWeight(SNetSgAverage):
    def __init__(self, options, session, syn_relations):
        super(SNetSgAvReNoWeight, self).__init__(options, session, syn_relations)

    def forward(self, examples, labels):
        syn_relations = self._session.run(self._syn_relations)

        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.

        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")
        self._emb = emb

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # ids for synonyms of the examples: [batch_size, relations_num]
        synonyms = tf.nn.embedding_lookup(self._syn_relations, examples)
        # flattern synonyms so we can get the embeddings of them
        synonyms_flat = tf.reshape(synonyms, [-1])
        # synonyms_emb: [batch_size*relations_num, emb_dim]
        synonyms_emb = tf.nn.embedding_lookup(emb, synonyms_flat)

        # get weighted average of the embedding of synonyms
        # change synonyms_emb to synonyms_emb_reshape:[batch_size, relations_num, emb_dim]
        synonyms_emb_reshaped = tf.reshape(synonyms_emb, [opts.batch_size, opts.relations_num, opts.emb_dim])

        # the 1st method getting the embedding
        synonyms_emb_reshaped_mean = tf.reduce_mean(synonyms_emb_reshaped, 1)

        synonyms_emb_reshaped_mean = tf.check_numerics(synonyms_emb_reshaped_mean,
                                                                  "syn_emb_reshaped_weighted_mean nan or inf")

        example_emb_syn_mean = tf.reduce_mean(
            tf.reshape(
                tf.concat(1, [example_emb, synonyms_emb_reshaped_mean]),
                [opts.batch_size, 2, opts.emb_dim]
            ), 1
        )
        example_emb_syn_mean = tf.check_numerics(example_emb_syn_mean, "emb_syn_mean nan or inf")

        # the 2nd method getting the mean embedding of the synonyms
        # sometimes diverges.  do not know why.
        # example_emb_syn = tf.concat(1,
        #                             [tf.expand_dims(example_emb, 1),
        #                              synonyms_emb_reshaped_weighted])
        # example_emb_syn_mean = tf.reduce_mean(example_emb_syn, 1)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(example_emb_syn_mean, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb_syn_mean,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        true_logits = tf.check_numerics(true_logits, "true_logits nan")
        sampled_logits = tf.check_numerics(sampled_logits, "sampled_logits nan")
        return true_logits, sampled_logits

    def get_final_emb(self):
        # get the averaged embeddings of the input and its related words as the output embeddings
        opts = self._options
        synonyms_flat = tf.reshape(self._syn_relations, [-1])
        synonyms_emb = tf.nn.embedding_lookup(self._emb, synonyms_flat)
        synonyms_emb_reshaped = tf.reshape(synonyms_emb,
                                           [opts.vocab_size, opts.relations_num, opts.emb_dim])
        synonyms_emb_reshaped_mean = tf.reduce_mean(synonyms_emb_reshaped, 1)
        emb_syn_mean = tf.reduce_mean(
            tf.reshape(
                tf.concat(1, [self._emb, synonyms_emb_reshaped_mean]),
                [opts.vocab_size, 2, opts.emb_dim]
            ), 1
        )
        return emb_syn_mean


class SNetSgSum(Word2Vec):
    def __init__(self, options, session, syn_relations):
        self._syn_relations = tf.constant(syn_relations)
        super(SNetSgSum, self).__init__(options, session)

    def forward(self, examples, labels):
        syn_relations = self._session.run(self._syn_relations)

        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.

        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")
        self._emb = emb

        # Word - related word: [vocab_size, relations_num]
        wrw = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.relations_num], -init_width, init_width),
            name="wrw")
        self._wrw = wrw

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # ids for synonyms of the examples: [batch_size, relations_num]
        synonyms = tf.nn.embedding_lookup(self._syn_relations, examples)
        # flattern synonyms so we can get the embeddings of them
        synonyms_flat = tf.reshape(synonyms, [-1])
        # synonyms_emb: [batch_size*relations_num, emb_dim]
        synonyms_emb = tf.nn.embedding_lookup(emb, synonyms_flat)

        # get weighted average of the embedding of synonyms
        # TODO: this is one of the ideas. remember to try:
        # TODO: 1.use trusted  paraphrase database
        # TODO: 2.use a softmax with biased threshold to train the synonym-layer to
        # TODO: trusted  paraphrase database
        # change synonyms_emb to synonyms_emb_reshape:[batch_size, relations_num, emb_dim]
        synonyms_emb_reshaped = tf.reshape(synonyms_emb,
                                          [opts.batch_size, opts.relations_num, opts.emb_dim])
        # weight the synonyms embeddings [batch_size, relations_num, emb_dim]
        wrw_weights = tf.nn.embedding_lookup(wrw, examples)
        wrw_weights = tf.expand_dims(wrw_weights, 2)
        synonyms_emb_reshaped_weighted = tf.mul(synonyms_emb_reshaped, wrw_weights)

        # the 1st method getting the embedding
        synonyms_emb_reshaped_weighted_mean = tf.reduce_sum(synonyms_emb_reshaped_weighted, 1)
        example_emb_syn_mean = tf.reduce_sum(
            tf.reshape(
                tf.concat(1, [example_emb, synonyms_emb_reshaped_weighted_mean]),
                [opts.batch_size, 2, opts.emb_dim]
            ), 1
        )


        # the 2nd method getting the mean embedding of the synonyms
        # sometimes diverges. do not know why.
        # example_emb_syn = tf.concat(1,
        #                             [tf.expand_dims(example_emb, 1),
        #                              synonyms_emb_reshaped_weighted])
        # example_emb_syn_mean = tf.reduce_mean(example_emb_syn, 1)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(example_emb_syn_mean, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb_syn_mean,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def get_final_emb(self):
        # get the averaged embeddings of the input and its related words as the output embeddings
        opts = self._options
        synonyms_flat = tf.reshape(self._syn_relations, [-1])
        synonyms_emb = tf.nn.embedding_lookup(self._emb, synonyms_flat)
        synonyms_emb_reshaped = tf.reshape(synonyms_emb,
                                           [opts.vocab_size, opts.relations_num, opts.emb_dim])
        wrw_weights = tf.expand_dims(self._wrw, 2)
        synonyms_emb_reshaped_weighted = tf.mul(synonyms_emb_reshaped, wrw_weights)
        synonyms_emb_reshaped_weighted_mean = tf.reduce_sum(synonyms_emb_reshaped_weighted, 1)
        emb_syn_mean = tf.reduce_sum(
            tf.reshape(
                tf.concat(1, [self._emb, synonyms_emb_reshaped_weighted_mean]),
                [opts.vocab_size, 2, opts.emb_dim]
            ), 1
        )
        return emb_syn_mean


    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        emb = self.get_final_emb()
        nemb = tf.nn.l2_normalize(emb, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, self._options.vocab_size))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx


class SNetSgAverageBias(SNetSgAverage):

    def __init__(self, options, session, syn_relations):
        super(SNetSgAverageBias, self).__init__(options, session, syn_relations)

    def forward(self, examples, labels):

        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.

        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")
        self._emb = emb

        # Word - related word weight: [vocab_size, relations_num]
        wrw = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.relations_num], -init_width, init_width),
            name="wrw")
        self._wrw = wrw

        # Word - related word bias: [vocab_size, relations_num]
        wrw_b = tf.Variable(tf.random_uniform([opts.vocab_size, opts.relations_num], -init_width, init_width),
                            name="wrw_b")
        self._wrw_b = wrw_b

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # ids for synonyms of the examples: [batch_size, relations_num]
        synonyms = tf.nn.embedding_lookup(self._syn_relations, examples)
        # flattern synonyms so we can get the embeddings of them
        synonyms_flat = tf.reshape(synonyms, [-1])
        # synonyms_emb: [batch_size*relations_num, emb_dim]
        synonyms_emb = tf.nn.embedding_lookup(emb, synonyms_flat) + 1e-10

        synonyms_emb = tf.check_numerics(synonyms_emb, "synonyms_emb")

        # get weighted average +bias = something like softmax of the embedding of synonyms
        # change synonyms_emb to synonyms_emb_reshape:[batch_size, relations_num, emb_dim]
        synonyms_emb_reshaped = tf.reshape(synonyms_emb, [opts.batch_size, opts.relations_num, opts.emb_dim])
        # synonyms_emb_reshaped = tf.check_numerics(synonyms_emb_reshaped, "synonyms_emb_reshaped")

        # weight the synonyms embeddings [batch_size, relations_num, emb_dim]
        wrw_weights = tf.nn.embedding_lookup(wrw, examples) #[batch_size, relations_num]
        wrw_weights = tf.expand_dims(wrw_weights, 2)  #[batch_size, relations_num, 1]
        wrw_bias = tf.nn.embedding_lookup(wrw_b, examples) #[batch_size, relations_num]
        wrw_bias = tf.expand_dims(wrw_bias, 2) #[batch_size, relations_num, 1]
        synonyms_emb_reshaped_weighted = tf.mul(synonyms_emb_reshaped, wrw_weights) + wrw_bias
        # synonyms_emb_reshaped_weighted = tf.check_numerics(synonyms_emb_reshaped_weighted,
        #                                                    "synonyms_emb_reshaped_weighted")

        # get the mean embedding of the synonyms
        synonyms_emb_mean = tf.reduce_mean(synonyms_emb_reshaped_weighted, 1)
        synonyms_emb_mean = tf.check_numerics(synonyms_emb_mean, "synonyms_emb_mean")
        # get the mean embedding of the synonyms and the center example word
        example_emb_syn = tf.reduce_mean(
            tf.reshape(
                tf.concat(1, [example_emb, synonyms_emb_mean]),
                [opts.batch_size, 2, opts.emb_dim]
            ), 1
        )
        example_emb_syn = tf.check_numerics(example_emb_syn, "synonyms_emb_syn")

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(example_emb_syn, true_w), 1) + true_b
        # true_logits = tf.check_numerics(true_logits, "true_logits")

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
        sampled_logits = tf.matmul(example_emb_syn,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        # sampled_logits = tf.check_numerics(sampled_logits, "sampled_logits")
        return true_logits, sampled_logits

    def get_final_emb(self):
        # get the averaged embeddings of the input and its related words as the output embeddings
        opts = self._options
        synonyms_flat = tf.reshape(self._syn_relations, [-1])
        synonyms_emb = tf.nn.embedding_lookup(self._emb, synonyms_flat)
        synonyms_emb_reshaped = tf.reshape(synonyms_emb,
                                           [opts.vocab_size, opts.relations_num, opts.emb_dim])
        wrw_weights = tf.expand_dims(self._wrw, 2)
        wrw_bias = tf.expand_dims(self._wrw_b, 2)
        synonyms_emb_reshaped_weighted = tf.mul(synonyms_emb_reshaped, wrw_weights) + wrw_bias
        synonyms_emb_reshaped_weighted_mean = tf.reduce_mean(synonyms_emb_reshaped_weighted, 1)
        emb_syn_mean = tf.reduce_mean(
            tf.reshape(
                tf.concat(1, [self._emb, synonyms_emb_reshaped_weighted_mean]),
                [opts.vocab_size, 2, opts.emb_dim]
            ), 1
        )
        return emb_syn_mean

def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
    """Train a word2vec model."""
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    opts.relations_num = 5
    syn_relations = WordnetNet(opts.relations_num).synsets_relations
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = SNetSgAverageBias(opts, session, syn_relations)
            model.read_analogies() # Read analogy questions
        for _ in xrange(opts.epochs_to_train):
            model.train()  # Process one epoch
            model.eval()  # Eval analogies.
        # Perform a final save.
        model.saver.save(session,
                         os.path.join(opts.save_path, "model.ckpt"),
                         global_step=model.global_step)
        if FLAGS.interactive:
            # E.g.,
            # [0]: model.analogy(b'france', b'paris', b'russia')
            # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
            _start_shell(locals())


if __name__ == "__main__":
    tf.app.run()

