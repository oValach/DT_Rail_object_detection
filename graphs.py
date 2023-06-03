import matplotlib.pyplot as plt

with open('D:\Work\DP\models//log_30_0.02.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    values = []
    for line in lines:
        values.append(str(line)[-6:-1])

test_data = []
train_data = []

for i,value in enumerate(values):
    if i % 2 == 0:
        train_data.append(float(value))
    else:
        test_data.append(float(value))

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

plt.style.use('bmh')
plt.plot(x, train_data)
plt.title('Train loss')
#plt.xticks(x)
plt.show()

plt.plot(x, test_data)
plt.title('Test loss')
#plt.xticks(x)
plt.show()
