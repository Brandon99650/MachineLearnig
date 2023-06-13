import numpy as np
import matplotlib.pyplot as plt

#random
def rand_samples(m, b, n_points, rand):
    x_coors ,y_coors,labels,= np.array([]), np.array([]), np.array([]) 
    c = 10 if m >= 0 else -10

    p_num = int(n_points / 2)
    neg_num = n_points - p_num
    for state, n_points in [['pos', p_num], ['neg', neg_num]]:
        x = np.random.randint(0, rand, n_points)
        r = np.random.randint(1, rand, n_points)

        if state == 'neg':
            y = m * x + b + (r * c)
            labels = np.append(labels, np.ones(n_points, dtype=int))
        else:
            y = m * x + b - (r * c)
            labels = np.append(labels, -1*np.ones(n_points, dtype=int))

        x_coors=np.append(x_coors, x)    
        y_coors=np.append(y_coors, y)    

    return x_coors, y_coors, labels
if __name__ == '__main__':
    w1, w0 = 20, 10
    #w1=m,w0=b

    n_points = 30
    rand = 30
    p_num = int(n_points / 2)

    x = np.arange(rand + 1)  
    y = w1* x + w0
    plt.plot(x, y)

    # randomly generate points
    x_coors, y_coors, labels = rand_samples(w1, w0, n_points, rand)

    # plot random points. Blue: positive, red: negative
    plt.plot(x_coors[:p_num], y_coors[:p_num], 'o', color='green')   # positive
    plt.plot(x_coors[p_num:], y_coors[p_num:], 'o', color='red')    # negative
    plt.show()
    m=np.array([])
for i in range(n_points):
        m=np.append(m,[x_coors[i],y_coors[i],labels[i]])
print (m)