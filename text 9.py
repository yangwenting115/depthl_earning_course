import tensorflow as tf 
	import numpy as np
	a = tf.constant([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
	b = tf.constant([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])
	# n = np.array(a)
	# 这里n=10
	mul1 = tf.multiply(a,b)
	mul2 = tf.multiply(a,a)
	w = (10*(tf.reduce_sum(mul1))-tf.reduce_sum(a)*tf.reduce_sum(b))/(10*(tf.reduce_sum(mul2))-tf.reduce_sum(a)*tf.reduce_sum(a))
	b = (tf.reduce_sum(b)-w*tf.reduce_sum(a))/10
	print(w,b)
