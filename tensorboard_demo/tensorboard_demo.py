# /usr/bin/python3
#tensorboard scalar:tf.summary.scalar
#tensorboard graph:tf.summary.FileWriter

#test:tensorboard --logdir=./log

import tensorflow as tf

sess = tf.Session()

with tf.name_scope("Wa_add_b"):
    a = tf.placeholder(dtype=tf.float32)
    b = tf.Variable([1.0],dtype=tf.float32)
    W = tf.Variable([1,2],dtype=tf.float32)
    addAB = W * a + b
    #step:1
    tf.summary.scalar("wight_max",tf.reduce_mean(W))
    tf.summary.scalar("b_value",tf.reduce_mean(b))
    tf.summary.scalar("a_value",tf.reduce_mean(a))
    #step:2
    merged = tf.summary.merge_all()
    train_summary = tf.summary.FileWriter('./log/',sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        tfmensumm,myadd = sess.run([merged,addAB],feed_dict={a:i})
        print(myadd)
        #step:3
        train_summary.add_summary(tfmensumm,i)
    train_summary.close()