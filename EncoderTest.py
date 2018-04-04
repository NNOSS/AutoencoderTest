

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

merged = tf.summary.merge_all()

with tf.Session() as sess:
	

    saver = tf.train.import_meta_graph('./Model/Mnist-Encoding.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./Model/'))

    graph = tf.get_default_graph()
    namelist = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print(namelist)
    decoder_in = graph.get_tensor_by_name("u_net/hidden_encode:0")
    #final = graph.get_tensor_by_name("u_net/uconv1:0")

    while(True):
		in_ind = raw_input("pick a valid integer: ")
		try:
			in_ind = int(in_ind)
		except:
			continue
		test = np.zeros(12)
		test[in_ind] = 1;
		feed_dict = {encoding: test}
		print(sess.run(final,feed_dict))