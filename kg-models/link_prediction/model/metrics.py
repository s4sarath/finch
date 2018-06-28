from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time


def s_next_batch(eval_triples,
                 entity_to_idx,
                 predicate_to_idx,
                 nb_entities,
                 batch_size):
    for _i, (s, p, o) in tqdm(enumerate(eval_triples), total=len(eval_triples), ncols=70):
        s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]
        xs = np.arange(nb_entities)
        xp = np.full(shape=(nb_entities,), fill_value=p_idx, dtype=np.int32)
        xo = np.full(shape=(nb_entities,), fill_value=o_idx, dtype=np.int32)
        for i in range(0, len(xs), batch_size):
            yield xs[i: i+batch_size], xp[i: i+batch_size], xo[i: i+batch_size]


def o_next_batch(eval_triples,
                   entity_to_idx, 
                   predicate_to_idx,
                   nb_entities,
                   batch_size):
    for _i, (s, p, o) in tqdm(enumerate(eval_triples), total=len(eval_triples), ncols=70):
        s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]
        xs = np.full(shape=(nb_entities,), fill_value=s_idx, dtype=np.int32)
        xp = np.full(shape=(nb_entities,), fill_value=p_idx, dtype=np.int32)
        xo = np.arange(nb_entities)
        for i in range(0, len(xs), batch_size):
            yield xs[i: i+batch_size], xp[i: i+batch_size], xo[i: i+batch_size]


def s_input_fn(eval_triples,
               entity_to_idx, 
               predicate_to_idx,
               nb_entities,
               batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: s_next_batch(eval_triples,
                             entity_to_idx, 
                             predicate_to_idx,
                             nb_entities,
                             batch_size),
        (tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([None,]),
         tf.TensorShape([None,]),
         tf.TensorShape([None,]),))
    iterator = dataset.make_one_shot_iterator()
    s, p, o = iterator.get_next()
    return {'s': s, 'p': p, 'o': o}


def o_input_fn(eval_triples,
               entity_to_idx, 
               predicate_to_idx,
               nb_entities,
               batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: o_next_batch(eval_triples,
                             entity_to_idx, 
                             predicate_to_idx,
                             nb_entities,
                             batch_size),
        (tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([None,]),
         tf.TensorShape([None,]),
         tf.TensorShape([None,]),))
    iterator = dataset.make_one_shot_iterator()
    s, p, o = iterator.get_next()
    return {'s': s, 'p': p, 'o': o}


def evaluate_rank(model,
                  valid_triples,
                  test_triples,
                  all_triples,
                  entity_to_idx,
                  predicate_to_idx,
                  nb_entities,
                  batch_size):

    for eval_name, eval_triples in [('valid', valid_triples), ('test', test_triples)]:
        
        _scores_s = np.fromiter(model.predict(
            lambda: s_input_fn(eval_triples,
                               entity_to_idx, 
                               predicate_to_idx,
                               nb_entities,
                               batch_size)),
            dtype=np.float32,
            count=len(eval_triples)*nb_entities)
        
        _scores_o = np.fromiter(model.predict(
            lambda: o_input_fn(eval_triples,
                               entity_to_idx, 
                               predicate_to_idx,
                               nb_entities,
                               batch_size)),
            dtype=np.float32,
            count=len(eval_triples)*nb_entities)

        ScoresS = _scores_s.reshape([len(eval_triples), nb_entities])
        ScoresO = _scores_o.reshape([len(eval_triples), nb_entities])

        ranks_s, ranks_o = [], []
        filtered_ranks_s, filtered_ranks_o = [], []

        for _i, ((s, p, o), scores_s, scores_o) in tqdm(enumerate(zip(eval_triples,
                                                                      ScoresS,
                                                                      ScoresO)),
                                                        total=len(eval_triples),
                                                        ncols=70):
            s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]

            ranks_s += [1 + np.argsort(np.argsort(- scores_s))[s_idx]]
            ranks_o += [1 + np.argsort(np.argsort(- scores_o))[o_idx]]

            filtered_scores_s = scores_s.copy()
            filtered_scores_o = scores_o.copy()

            rm_idx_s = [entity_to_idx[fs] for (fs, fp, fo) in all_triples if fs != s and fp == p and fo == o]
            rm_idx_o = [entity_to_idx[fo] for (fs, fp, fo) in all_triples if fs == s and fp == p and fo != o]

            filtered_scores_s[rm_idx_s] = - np.inf
            filtered_scores_o[rm_idx_o] = - np.inf

            filtered_ranks_s += [1 + np.argsort(np.argsort(- filtered_scores_s))[s_idx]]
            filtered_ranks_o += [1 + np.argsort(np.argsort(- filtered_scores_o))[o_idx]]

        ranks = ranks_s + ranks_o
        filtered_ranks = filtered_ranks_s + filtered_ranks_o

        for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
            mean_rank = np.mean(setting_ranks)
            print('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
            for k in [1, 3, 5, 10]:
                hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                print('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
