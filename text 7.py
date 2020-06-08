import tensorflow as tf 
	import matplotlib.pyplot as plt 
	import numpy as np
	mnist =tf.keras.datasets.mnist
	(train_x,train_y),(test_x,test_y) = mnist.load_data()
	plt.rcParams['font.sans-serif']="SimHei"
	plt.figure(figsize=(10,10))
	
	for i in range(16):
	    num = np.random.randint(1,60000)
	    plt.subplot(4,4,i+1)
	    plt.axis("off")
	    plt.imshow(train_x[num],cmap='gray')
	    plt.title(train_y[num])
	
	plt.suptitle("MNIST测试集样本",fontsize="20",color="red")
	plt.show()
