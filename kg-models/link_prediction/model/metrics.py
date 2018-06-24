from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time


def subj_next_batch(eval_triples,
                    entity_to_idx,
                    predicate_to_idx,
                    nb_entities):
    for _i, (s, p, o) in tqdm(enumerate(eval_triples), total=len(eval_triples), ncols=70):
        s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]
        xs = np.arange(nb_entities)
        xp = np.full(shape=(nb_entities,), fill_value=p_idx, dtype=np.int32)
        xo = np.full(shape=(nb_entities,), fill_value=o_idx, dtype=np.int32)
        yield xs, xp, xo


def obj_next_batch(eval_triples,
                   entity_to_idx, 
                   predicate_to_idx,
                   nb_entities):
    for _i, (s, p, o) in tqdm(enumerate(eval_triples), total=len(eval_triples), ncols=70):
        s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]
        xs = np.full(shape=(nb_entities,), fill_value=s_idx, dtype=np.int32)
        xp = np.full(shape=(nb_entities,), fill_value=p_idx, dtype=np.int32)
        xo = np.arange(nb_entities)
        yield xs, xp, xo


def subj_input_fn(eval_triples,
                  entity_to_idx, 
                  predicate_to_idx,
                  nb_entities):
    dataset = tf.data.Dataset.from_generator(
        lambda: subj_next_batch(eval_triples,
                                entity_to_idx, 
                                predicate_to_idx,
                                nb_entities),
        (tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([nb_entities]),
         tf.TensorShape([nb_entities]),
         tf.TensorShape([nb_entities]),))
    iterator = dataset.make_one_shot_iterator()
    s, p, o = iterator.get_next()
    return {'s': s, 'p': p, 'o': o}


def obj_input_fn(eval_triples,
                 entity_to_idx, 
                 predicate_to_idx,
                 nb_entities):
    dataset = tf.data.Dataset.from_generator(
        lambda: obj_next_batch(eval_triples,
                               entity_to_idx, 
                               predicate_to_idx,
                               nb_entities),
        (tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([nb_entities]),
         tf.TensorShape([nb_entities]),
         tf.TensorShape([nb_entities]),))
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
        
        t0 = time.time()
        _scores_subj = list(model.predict(
            lambda: subj_input_fn(eval_triples,
                                  entity_to_idx, 
                                  predicate_to_idx,
                                  nb_entities),
            yield_single_examples = False))
        print('%.1f secs'%(time.time() - t0))
        
        t0 = time.time()
        _scores_obj = list(model.predict(
            lambda: obj_input_fn(eval_triples,
                                 entity_to_idx, 
                                 predicate_to_idx,
                                 nb_entities),
            yield_single_examples = False))
        print('%.1f secs'%(time.time() - t0))

        Scores_subj = np.reshape(_scores_subj, [len(eval_triples), nb_entities])
        Scores_obj = np.reshape(_scores_obj, [len(eval_triples), nb_entities])

        ranks_subj, ranks_obj = [], []
        filtered_ranks_subj, filtered_ranks_obj = [], []

        for _i, ((s, p, o), scores_subj, scores_obj) in tqdm(enumerate(zip(eval_triples,
                                                                           Scores_subj,
                                                                           Scores_obj)),
                                                             total=len(eval_triples),
                                                             ncols=70):
            s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]

            ranks_subj += [1 + np.argsort(np.argsort(- scores_subj))[s_idx]]
            ranks_obj += [1 + np.argsort(np.argsort(- scores_obj))[o_idx]]

            filtered_scores_subj = scores_subj.copy()
            filtered_scores_obj = scores_obj.copy()

            rm_idx_s = [entity_to_idx[fs] for (fs, fp, fo) in all_triples if fs != s and fp == p and fo == o]
            rm_idx_o = [entity_to_idx[fo] for (fs, fp, fo) in all_triples if fs == s and fp == p and fo != o]

            filtered_scores_subj[rm_idx_s] = - np.inf
            filtered_scores_obj[rm_idx_o] = - np.inf

            filtered_ranks_subj += [1 + np.argsort(np.argsort(- filtered_scores_subj))[s_idx]]
            filtered_ranks_obj += [1 + np.argsort(np.argsort(- filtered_scores_obj))[o_idx]]

        ranks = ranks_subj + ranks_obj
        filtered_ranks = filtered_ranks_subj + filtered_ranks_obj

        for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
            mean_rank = np.mean(setting_ranks)
            print('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
            for k in [1, 3, 5, 10]:
                hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                print('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
