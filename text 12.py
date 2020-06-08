import matplotlib.pyplot as plt
	import numpy as np
	import tensorflow as tf
	
	boston_house = tf.keras.datasets.boston_housing
	(train_x,train_y),(test_x,test_y) = boston_house.load_data()
	
	train_x.shape,train_y.shape
	test_x.shape,test_y.shape
	
	num_train = len(train_x)
	num_test = len(test_x)
	
	x_train = (train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
	y_train = train_y
	
	x_test = (test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))
	y_test = test_y
	
	x0_train = np.ones(num_train).reshape(-1,1)
	x0_test = np.ones(num_test).reshape(-1,1)
	
	X_train = tf.cast(tf.concat([x0_train,x_train],axis=1),tf.float32)
	X_test = tf.cast(tf.concat([x0_test,x_test],axis=1),tf.float32)
	
	X_train.shape,X_test.shape
	
	Y_train = tf.constant(y_train.reshape(-1,1),tf.float32)
	Y_test  = tf.constant(y_test.reshape(-1,1),tf.float32)
	
	Y_train.shape,Y_test.shape
	
	
	
	learn_rate = 0.01
	iter = 2000
	display_step = 100
	
	np.random.seed(612)
	W = tf.Variable(np.random.randn(14,1),dtype=tf.float32)
	
	mse_train = []
	mse_test = []
	
	for i in range(0,iter+1):
	    with tf.GradientTape() as tape:
	
	        PRED_train = tf.matmul(X_train,W)
	        Loss_train = 0.5*tf.reduce_mean(tf.square(Y_train-PRED_train))
	
	        PRED_test = tf.matmul(X_test,W)
	        Loss_test = 0.5*tf.reduce_mean(tf.square(Y_test-PRED_test))
	
	    mse_train.append(Loss_train)
	    mse_test.append(Loss_test)
	
	    dL_dw=tape.gradient(Loss_train,W)
	    W.assign_sub(learn_rate*dL_dw)
	
	    
	    if i % display_step ==0:
	        print("i:%i,Train Loss: %f,Test Loss:%f" % (i,Loss_train,Loss_test))
	
	plt.figure(figsize=(20,4))
	plt.subplot(131)
	plt.ylabel("MSE")
	plt.plot(mse_train,color="blue",linewidth=3)
	plt.plot(mse_test,color="red",linewidth=1.5)
	
	plt.subplot(132)
	plt.ylabel("Pirce")
	plt.plot(y_train,color="blue",linewidth=3,marker="o",label="true_pirce")
	plt.plot(PRED_train,color="red",linewidth=1.5,marker=".",label="predict")
	plt.legend()
	
	plt.subplot(133)
	plt.ylabel("Price")
	plt.plot(y_test,color="blue",linewidth=3,marker="o",label="true_pirce")
	plt.plot(PRED_test,color="red",linewidth=1.5,marker=".",label="predict")
	plt.legend()
	    
	plt.show()
