import tensorflow as tf

def soft_label(self, preds, labels, plimit=0.85, nlimit=0.15):
    # soft labels for positive-1.0 and negative-0.0 labels.
    minus = labels - preds
    pr = tf.where((minus < (1.0 - plimit)) & (minus > 0.0), tf.ones_like(minus), tf.zeros_like(minus))
    nr = tf.where((minus > (0.0 - nlimit)) & (minus < 0.0), tf.ones_like(minus), tf.zeros_like(minus))
    slabel = pr * preds + (labels - pr) + nr * preds
    return slabel