

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

#merged = tf.summary.merge_all()

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./Model2/Mnist-Encoding.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./Model2/'))

    graph = tf.get_default_graph()
    #namelist = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(namelist)
    decoder_in = graph.get_tensor_by_name("u_net/hidden_encode/Relu:0")
    final = graph.get_tensor_by_name("decoding:0")
    step = 0
    sum_writer = tf.summary.FileWriter("logs_enctest", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    while(True):
		in_ind = raw_input("pick a valid integer: ")
		try:
			in_ind = int(in_ind)
		except:
			continue
		test = np.zeros(25)
		test[in_ind] = 30;
		test = np.expand_dims(test, axis=0)
		feed_dict = {decoder_in: test}
		out = sess.run(final,feed_dict)
		sum_writer.add_summary(out, step)
		step += 1;