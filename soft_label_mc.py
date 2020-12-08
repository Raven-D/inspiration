import numpy as np
import tensorflow as tf

s = tf.Session()

def softlabel_mc(logits, labels, cls_num=5, rate=1.0, dtype=tf.float32):
    fcls_num = float(cls_num)
    if (rate <= 1.0):
        limit = (1.0 / fcls_num) + (1.0 / fcls_num) * -np.log(1./fcls_num)
    else:
        limit = (1.0 / fcls_num) * rate
    logits = tf.convert_to_tensor(logits, dtype=dtype)
    stm = tf.nn.softmax(logits, axis=-1)
    ohot = tf.one_hot(labels, cls_num)
    filt = tf.convert_to_tensor(ohot * limit, dtype=dtype)
    print('filt')
    print(s.run(filt))
    minus = filt - stm * ohot
    print('minus')
    print(s.run(minus))
    sum_minus = tf.reduce_sum(minus, axis=-1)
    print('resd')
    print(s.run(sum_minus))
    result = tf.where(sum_minus > 0.0, ohot, stm)
    print('result')
    print(s.run(result))
    return result


def ce_loss(logits, labels, dtype=tf.float32):
    logits = tf.convert_to_tensor(logits, dtype=dtype)
    labels = tf.convert_to_tensor(labels, dtype=dtype)
    stm = tf.nn.softmax(logits, axis=-1)
    return -tf.reduce_sum(labels * tf.log(stm), -1)

a = np.array([[1., 2., 3.], [5., 2., 1.], [2., 3., 1.]])
print('softmax')
print(s.run(tf.nn.softmax(a, -1)))

new_labels = softlabel_mc(a, [2, 0, 0], cls_num=3, rate=1.5)
print(s.run(ce_loss(a, new_labels)))