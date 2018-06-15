from configs.rnn_config import args

import tensorflow as tf
import numpy as np
import time
import pprint


def load_embedding():
    t0 = time.time()
    embedding = np.load('../data/files_processed/word_embedding.npy')
    print("Load word_embed: %.2fs"%(time.time()-t0))
    return embedding


def cell_fn(sz):
    if args.rnn_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(sz,
                                      kernel_initializer=tf.orthogonal_initializer())
    elif args.rnn_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(sz,
                                       initializer=tf.orthogonal_initializer())
    else:
        raise ValueError("args.rnn_type has to be 'lstm' or 'gru'")
    return cell


def mask_fn(x):
    return tf.sign(tf.reduce_sum(x, -1))


def rnn(x, cell_fw, cell_bw):
    seq_len = tf.count_nonzero(tf.reduce_sum(x, -1), 1)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                      cell_bw,
                                                      x,
                                                      seq_len,
                                                      dtype=tf.float32)
    outputs = tf.concat(outputs, -1)
    if args.rnn_type == 'lstm':
        states = tf.concat([s.h for s in states], -1)
    elif args.rnn_type == 'gru':
        states = tf.concat(states, -1)
    else:
        raise ValueError("args.rnn_type has to be 'lstm' or 'gru'")
    return outputs, states


def embed(x, embedding, is_training):
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.layers.dropout(x, 0.2, is_training)
    return x


def clip_grads(loss):
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    clipped_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)
    return zip(clipped_grads, params)


def attn(query, context, v, w1, w2, masks):
    query = tf.expand_dims(query, 1)
    keys = w1(context)
    values = w2(context)

    align = v * tf.tanh(query + keys)
    align = tf.reduce_sum(align, [2])

    paddings = tf.fill(tf.shape(align), float('-inf'))
    align = tf.where(tf.equal(masks, 0), paddings, align)

    align = tf.nn.softmax(align)
    align = tf.expand_dims(align, -1)
    val = tf.squeeze(tf.matmul(values, align, transpose_a=True), -1)
    return val


def k_max(x, k):
    x = tf.transpose(x, [0,2,1])
    x = tf.nn.top_k(x, k, sorted=True).values
    x = tf.reshape(x, [-1, k*args.hidden_units])
    return x


def forward(features, mode):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    x1, x2 = features['input1'], features['input2']
    
    embedding = tf.convert_to_tensor(load_embedding())
    x1 = embed(x1, embedding, is_training)
    x2 = embed(x2, embedding, is_training)
    mask1 = mask_fn(x1)
    mask2 = mask_fn(x2)

    cell_fw, cell_bw = cell_fn(args.hidden_units//2), cell_fn(args.hidden_units//2)
    o1, s1 = rnn(x1, cell_fw, cell_bw)
    o2, s2 = rnn(x2, cell_fw, cell_bw)
    
    with tf.variable_scope('attention'):
        v = tf.get_variable('attn_v', [args.hidden_units])
        w1 = tf.layers.Dense(args.hidden_units)
        w2 = tf.layers.Dense(args.hidden_units)
        attn1 = attn(s1, o2, v, w1, w2, mask2)
        attn2 = attn(s2, o1, v, w1, w2, mask1)

    k1 = k_max(o1, 3)
    k2 = k_max(o2, 3)

    x = tf.concat([
        tf.abs(k1 - k2),
        (k1 * k2),
        attn1,
        attn2,
        tf.abs(attn1 - attn2),
        (attn1 * attn2),
    ], -1)
    
    with tf.variable_scope('output'):
        x = tf.layers.dropout(x, 0.4, training=is_training)
        x = tf.layers.dense(x, 2*args.hidden_units, tf.nn.elu)
        x = tf.layers.dense(x, args.hidden_units, tf.nn.elu)
        x = tf.squeeze(tf.layers.dense(x, 1), -1)
    
    return x


def model_fn(features, labels, mode):
    logits = forward(features, mode)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.sigmoid(logits))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
        global_step = tf.train.get_global_step()

        LR = {'start': 1e-3, 'end': 5e-4, 'steps': 10000}
        
        lr_op = tf.train.exponential_decay(
            LR['start'], global_step, LR['steps'], LR['end']/LR['start'])
        
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.to_float(labels)))

        train_op = tf.train.AdamOptimizer(lr_op).apply_gradients(
            clip_grads(loss_op), global_step=global_step)

        lth = tf.train.LoggingTensorHook({'lr': lr_op}, every_n_iter=100)
        
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op, training_hooks=[lth])
