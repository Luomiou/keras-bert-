
# encoding = utf - 8


import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('C:\\Users\Administrator\Desktop\chinese_L-12_H-768_A-12\\bert_model.ckpt.data-00000-of-00001')
    saver.restore(sess,'./new/model.ckpt')
