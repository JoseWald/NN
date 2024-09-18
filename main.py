import numpy as np

x_enter=np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]), dtype=float)

#output value (0=red , 1=blue)
y=np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float)

x_enter=x_enter/np.max(x_enter,axis=0)

x=np.split(x_enter,[8])[0]
x_prediction=np.split(x_enter,[8])[1]

class Neural_network(object):
    def __init__(self):
        self.input_size=2
        self.output_size=1
        self.hidden_size=3

        self.w1=np.random.randn(self.input_size,self.hidden_size)
        self.w2=np.random.randn(self.hidden_size,self.output_size)

    def forward(self,X):
        self.z=np.dot(X,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        o=self.sigmoid(self.z3)
        return o
    
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    
NN=Neural_network()
n=NN.forward(x)
print("Output predicted by IA"+ str(n))
print("The right output should by:"+str(y))
