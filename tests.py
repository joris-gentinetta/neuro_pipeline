import numpy as np

a = np.array([[1,2], [1,3]])
b=a==1
print(np.sum(b, axis=0))