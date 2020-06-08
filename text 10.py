import numpy as np
	x1=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
	x2=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])
	y = np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
	# print(x1.shape,x2.shape,y.shape)
	x0=np.ones(len(x1))
	X=np.stack((x0,x1,x2),axis=1)
	# print(X)
	Y=np.array(y).reshape(-1,1)
	# print(Y)
	Xt =np.transpose(X)
	XtX_1 = np.linalg.inv(np.matmul(Xt,X))
	XtX_1_Xt = np.matmul(XtX_1,Xt)
	W=np.matmul(XtX_1_Xt,Y)
	# print(W)
	W = W.reshape(-1)
	print("请输入房屋面积（20-500）和房间数（1-10），预测房屋销售价格\n")
	x1 = float(input("面积:"))
	x2 = int(input("房间数:"))
	if x1<20 or x1>500:
	    print("房间面积不对，请重新输入")
	    x1 = float(input("面积:"))
	elif x2<1 or x2>10:
	    print("房间数目不对，请重新输入")
	    x2 = int(input("房间数:"))
	pred = W[1]*x1+W[2]*x2+W[0]
	print("预测的价格为：",round(pred,2),"万元")
