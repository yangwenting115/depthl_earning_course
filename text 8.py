import tensorflow as tf 
	a = tf.constant([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
	b = tf.constant([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])
	
	meana = tf.reduce_mean(a)
	meanb = tf.reduce_mean(b)
	
	a1 = a-meana
	b1 = b-meanb
	w = tf.reduce_sum(tf.multiply(a1,b1))/tf.reduce_sum(tf.multiply(a1,a1))
	b = meanb-w*meana
	print(w,b)
