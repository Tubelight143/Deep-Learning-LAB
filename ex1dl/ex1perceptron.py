import numpy as np
x1=np.array([0,0,1,1])
x2=np.array([0,1,0,1])
y=np.array([0,1,1,1])

epochs=int(input("Enter the epochs:"))

bias=1
l=0.4
a=[]

a.append(float(input("Enter the weights for bias:")))
a.append(float(input("Enter the weights for x1:")))
a.append(float(input("Enter the weights for x2:")))

w=np.array(a)
n=x1.shape[0]

for k in range(epochs):
    for i in range(n):
        f=x1[i]*w[1]+x2[i]*w[2]+bias*w[0]
        y_out=(f>0).astype("int")
        error=y[i]-y_out
        if error!=0:
            w[1]=w[1]+l*error*x1[i]
            w[2]=w[2]+l*error*x2[i]
            w[0]=w[0]+l*error*bias
        print("----------------")
        print("Updated weights after",i+1,"input instance")
        print("x1 weights",w[1])
        print("x2 weights",w[2])
        print("bias weights",w[0])
        print("----------------")
        print("Final weights after",epochs,"epochs(s)")
        print("Updates weights for x1:",w[1])
        print("Updates weights for x2:",w[2])
        print("Updates weights for bias:",w[0])
