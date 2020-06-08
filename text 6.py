import matplotlib.pyplot as plt 
	from PIL import Image
	
	plt.rcParams['font.sans-serif']="SimHei"
	img = Image.open("lena.tiff")
	img_r,img_g,img_b = img.split()
	plt.figure(figsize=(10,10))
	
	
	plt.subplot(221)
	plt.axis("off")
	img_small = img.resize((50,50))
	img_small = img_small.convert("I")
	plt.imshow(img_small)
	plt.title("R-缩放",fontsize=14)
	
	plt.subplot(222)
	img_2 = img.resize((250,250))
	img_2 = img_2.convert("I")
	img_2 = img_2.transpose(Image.FLIP_LEFT_RIGHT)
	img_2 = img_2.transpose(Image.ROTATE_270)
	plt.imshow(img_2)
	plt.title("G-镜像+旋转",fontsize=14)
	
	plt.subplot(223)
	plt.axis("off")
	img_3 = img.resize((250,250))
	img_3 = img_3.convert("I")
	img_3 = img_3.crop((0,0,150,150))
	plt.imshow(img_3)
	plt.title("B-裁剪",fontsize=14)
	
	img_rgb = Image.merge("RGB",[img_r,img_g,img_b])
	plt.subplot(224)
	plt.axis("off")
	plt.imshow(img_rgb)
	plt.title("RGB",fontsize=14)
	
	# img.save("tiff.png")
	plt.suptitle("图像基本操作",fontsize="20",color="blue")
	plt.show()
