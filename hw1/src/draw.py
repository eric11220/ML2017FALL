from matplotlib import pyplot as plt

lamb = [0.1, 0.01, 0.001, 0.0001]
test_all = [8.1, 6.534, 6.351, 6.339]
train_all = [6.480, 5.464, 5.387, 5.385]

test_pm = [8.496, 6.666, 6.437, 6.421]
train_pm = [6.989, 5.906, 5.828, 5.826]

'''
test_err, = plt.plot(range(4), test_all, label='testing error')
train_err, = plt.plot(range(4), train_all, label='training error')
'''

test_err, = plt.plot(range(4), test_pm, label='testing error')
train_err, = plt.plot(range(4), train_pm, label='training error')

plt.xticks(range(4), lamb, rotation='45')
plt.legend(handles=[test_err, train_err])
plt.show()
