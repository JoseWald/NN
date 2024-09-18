import numpy as np

x_enter=np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[1,1.5]), dtype=float)

#output value (0=red , 1=blue)
y=np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float)

x_enter=x_enter/np.max(x_enter,axis=0)

x=np.split(x_enter,[8])[0]

xPrediction=np.split(x_enter,[8])[1]

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
    
    #derivative of the function sigmoid
    def sigmoid_prime(self,s):
        return s*(1-s)
    
    def backward(self,x,y,o):
            self.o_error=y-o #input - output
            self.delta_error=self.o_error * self.sigmoid_prime(o)

            self.z2_error=self.delta_error.dot(self.w2.T)
            self.z2_delta=self.z2_error *self.sigmoid_prime(self.z2)

            self.w1+=x.T.dot(self.z2_delta)
            self.w2+=self.z2.T.dot(self.delta_error)

    def train(self,x,y):
         o=self.forward(x)
         self.backward(x,y,o)

    def predict(self):
         print("Predicted data after the training:")
         print("Input : \n" + str(xPrediction))
         print("Output : \n" + str(self.forward(xPrediction)))

         if(self.forward(xPrediction)<=0.5):
              print("It's a RED leaf")
         else:
              print("It's a BLUE leaf")

    
NN=Neural_network()


for i in range(300):
     print("#"+str(i))
     print("Input value:\n"+ str(x))
     print("actual output:\n"+ str(y))
     print("Predicted value :\n"+ str(np.round(NN.forward(x),2)))
     print("\n")
     NN.train(x,y)
  


NN.predict()
