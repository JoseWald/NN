import numpy as np

x_enter=np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]), dtype=float)

#output value (0=red , 1=blue)
y=np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float)

x_enter=x_enter/np.max(x_enter,axis=0)

print(x_enter)