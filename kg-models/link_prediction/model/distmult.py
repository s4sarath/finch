import tensorflow as tf
import numpy as np
import pprint


def forward(features, params):
    e_embed = tf.get_variable('e_embed',
                              [params['e_vocab_size'], params['embed_dim']],
                              initializer=tf.contrib.layers.xavier_initializer())
    p_embed = tf.get_variable('p_embed',
                              [params['p_vocab_size'], params['embed_dim']],
                              initializer=tf.contrib.layers.xavier_initializer())
    
    s = tf.nn.embedding_lookup(e_embed, features['s'])
    p = tf.nn.embedding_lookup(p_embed, features['p'])
    o = tf.nn.embedding_lookup(e_embed, features['o'])
    
    logits = tf.reduce_sum(s*p*o, axis=1)
    
    return logits
    
    
def model_fn(features, labels, mode, params):
    logits = forward(features, params)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
        tf.logging.info('params: %d'%count_train_params())
        
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                         labels=labels))
        
        train_op = tf.train.AdamOptimizer().minimize(
            loss_op, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode = mode,
                                          loss = loss_op,
                                          train_op = train_op)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions = tf.sigmoid(logits))


def count_train_params():
    return np.sum([np.prod([d.value for d in v.get_shape()]) for v in tf.trainable_variables()])
