from linearTrain import predict, linear_reg
import numpy as np

a = 3
b = 2
x = np.asarray([i for i in range(1, 11)])
y = a*x + b
x = x.reshape((-1, 1))

coef = linear_reg(x, y, n_epoch=1000, batch_size=10, lr=6e-3)
print(coef)
