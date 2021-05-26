import numpy as np

from operator import add
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import copy
from heapq import nsmallest

# ===================================================================
# read all digits data
def ParseData(files):
    digit = []
    data = []

    for filename in files:
        f = open(filename)
        line = f.readline() 
        while(line != ""):
            fstr = str.split(line)
            num = int(float(fstr[0]))
            img = []
            for i in range(16):
                row = []
                for j in range(16):
                    val = (i*16) + j
                    row.append(float(fstr[val+1]))
                img.append(row)
            digit.append(num)
            data.append(img)
            line = f.readline() 
        f.close()

    return digit, data
    
# ===================================================================
# first feature = total intensity
def CalcIntensity(arr):
    I = [np.sum(img) for img in arr]
    return I

# ===================================================================
# second feature = veritcal asymmetry
def CalcVSymmetry(arr):
    S = []
    for img in arr:
        nmf = np.flipud(img)         # flip the matrix up-down
        nps = np.absolute(np.subtract(img, nmf))
        S.append(np.sum(nps))
    return S

# ===================================================================
# shift and scale for normalizing data
def Shift_Scale(d_arr):
    shift = (max(d_arr) + min(d_arr))/2.0
    scale = (max(d_arr) - min(d_arr))/2.0
    return shift, scale

# ===================================================================
# normalize the data with shift & scale
def Normalize(d_arr, shift, scale):
    d_n = [(v-shift)/scale for v in d_arr]
    return d_n

# ===================================================================
# separate the 1s and non-1s (for scatter plots)
def separateOnes(train_d, train_I, train_S):
    I1 = [train_I[i] for i in range(len(train_d)) if train_d[i] == 1]
    S1 = [train_S[i] for i in range(len(train_d)) if train_d[i] == 1]

    Ix = [train_I[i] for i in range(len(train_d)) if train_d[i] != 1]
    Sx = [train_S[i] for i in range(len(train_d)) if train_d[i] != 1]
    return I1, S1, Ix, Sx

class NeuralNetwork:
    def __init__(self):

        # 2inputs, 10 hidden layers, 1 output node
        self.wij   = np.random.rand(2,10) # input to hidden layer weights
        self.wjk   = np.random.rand(10,1) # hidden layer to output weights
        
    def tanh(self, x, w):
        z = np.dot(x, w)
        return np.tanh(z)
    
    def tanh_derivative(self, x, w):
        return 1.0 - (self.tanh(x,w)*self.tanh(x, w))
    
    def gradient_descent(self, x, y, iterations):
        Ein = []
        for i in range(iterations):
            Xi = x
            Xj = self.tanh(Xi, self.wij)
            yhat = self.tanh(Xj, self.wjk)
            # gradients for hidden to output weights
            g_wjk = np.dot(Xj.T, (y - yhat) * self.tanh_derivative(Xj, self.wjk))

            Ein.append( np.sum(np.multiply( (y - np.sign(yhat)), (y - np.sign(yhat))))/len(y))

            # gradients for input to hidden weights
            g_wij = np.dot(Xi.T, np.dot((y - yhat) * self.tanh_derivative(Xj, self.wjk), self.wjk.T) * self.tanh_derivative(Xi, self.wij))
            # update weights
            self.wij += (0.01)*g_wij
            self.wjk += (0.01)*g_wjk

        return(Ein)

    def Predict(self, Xi):
        Xj = self.tanh(Xi, self.wij)
        yhat = np.sign(self.tanh(Xj, self.wjk))
        return (yhat.tolist())

if __name__=="__main__":

    # ===================================================================
    # parse and store all digits from all files (train and test combined)
    files = []
    files.append("ZipDigits.train")
    files.append("ZipDigits.test")
    digit, data = ParseData(files)

    # ===================================================================
    # features used = Intensity and Vertical ASymmetry
    # calculate intensity and store it in I
    # calculate Vertical ASymmetry and store it in S
    # create data features for data in d
    I = CalcIntensity(data)
    S = CalcVSymmetry(data)

    # ===================================================================
    # normalize intensity for all numbers
    # shift = ( (min(values)) + (max(values)) )/ 2
    # scale = ( abs(min(values)) + abs(max(values)) )/ 2
    shift, scale = Shift_Scale(I)
    I_n = Normalize(I, shift, scale)

    # ===================================================================
    # normalize vertical Asymmetry for all numbers
    # shift = ( (min(values)) + (max(values)) )/ 2
    # scale = ( abs(min(values)) + abs(max(values)) )/ 2
    shift, scale = Shift_Scale(S)
    S_n = Normalize(S, shift, scale)

    # ===================================================================
    # create train (300 random), and test (remaining)
    # generate 300 rand int between 0 and N (N = # of data points)
    # these are the index for train data, remaining are test data

    random.seed(75)
    N = len(digit)
    R = random.sample(range(0, N), 300)

    train_d = [digit[i] for i in R]
    train_I = [I_n[i] for i in R]
    train_S = [S_n[i] for i in R]

    # create the Ys for the training records
    Y = np.zeros((len(train_d), 1))
    for row in range(len(train_d)):
        if (train_d[row] == 1):
            Y[row] = 1
        else:
            Y[row] = -1

    # create the training data points the way we need it
    Dx = []
    for i in range(len(train_d)):
        Dx.append([train_I[i], train_S[i]])

    Dy = np.zeros((300,1))
    for i in range(len(train_d)):
        Dy[i] = Y[i]


    np.random.seed(75)
    neural_network = NeuralNetwork()

    X = np.array(Dx)
    y = Dy
    Ein = neural_network.gradient_descent(X, y, 20000)
    
    # ============================================================================================
    # create framework for graphs and plots [3 graphs with subplots inside]. for Q2 and Q3
    fig, (ax1) = plt.subplots(2, 2)
    plt.tight_layout()
    ax1[0][0].grid()
    ax1[1][0].grid()
    ax1[1][1].grid()

    # graph Ein Vs Iterations
    xval = np.arange(1, 20000, 10)
    yval = []
    for p in xval:
        yval.append(Ein[p])
        
    ax1[0][0].plot(xval, yval, marker='.', color="r", label='Ein')
    ax1[0][0].set_xlabel("Iterations")
    ax1[0][0].set_ylabel("Ein")
    ax1[0][0].set_title("Ein Vs. Iterations")

    # print(neural_network.Predict([[0.1,0.2], [.2, -0.3]]))

    # ============================================================================================
    # decision boundary creation for k related to min(e_cv) for training data
    xlist = np.linspace(-1, 1, 200)
    ylist = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xlist, ylist)

    Z=[]
    for x_row, y_row in zip(X,Y):
        z_row=zip(x_row,y_row)
        Z.append(z_row)    

    nn=[]
    for z_row in Z:
        #transform z_row and predict
        row=[neural_network.Predict(pt)[0] for pt in z_row]
        nn.append(row)

    # ============================================================================================
    # plot decision boundary training data
    # separate out the 1s and non-1s - training data
    I1, S1, Ix, Sx = separateOnes(train_d, train_I, train_S)
    cmap = colors.ListedColormap(['#dddddd', '#66ffff'])
    cp = ax1[1][0].contour(X, Y, nn, [0])
    ax1[1][0].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax1[1][0].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    # fig.colorbar(cp, ax=ax1[1][0])
    ax1[1][0].set_xlabel("Total Intensity")
    ax1[1][0].set_ylabel("Vertical ASymmetry")
    ax1[1][0].legend()
    ax1[1][0].set_title("Training Decision Boundary (12.2ab), 1s and non-1s")

    plt.show()

"""
    # ============================================================================================
    # create the test records and Ys for the test records

    test_d = [digit[i] for i in range(N) if i not in R]
    test_I = [I_n[i] for i in range(N) if i not in R]
    test_S = [S_n[i] for i in range(N) if i not in R]

    # create the Ys for the test records
    Yt = np.zeros((len(test_d), 1))
    for row in range(len(test_d)):
        if (test_d[row] == 1):
            Yt[row] = 1
        else:
            Yt[row] = -1

    # create the test data points the way we need it for KNN [ ([x1, x2], Y) ]
    Dtest = []
    for i in range(len(test_d)):
        Dtest.append( ([test_I[i], test_S[i]], Yt[i]) )

    # ============================================================================================
    # create framework for graphs and plots [3 graphs with subplots inside]. for Q2 and Q3
    fig, (ax1) = plt.subplots(2, 2)
    plt.tight_layout()
    ax1[0][0].grid()
    ax1[1][0].grid()
    ax1[1][1].grid()


    # ============================================================================================
    # plot K vs. Ecv (on training data)

    kmin = [k_range[i-1] for i in k_range if e_cv[i-1] == min(e_cv)]

    ax1[0][0].plot(k_range, e_cv, marker='.', color="k", label='Ecv')
    ax1[0][0].scatter(kmin[0], min(e_cv), marker='.', s=200, color="r", label='Min Ecv')
    ax1[0][0].set_xlabel("k values")
    ax1[0][0].set_ylabel("Ecv")

    c_str = "Min Ecv = " + str(np.around(min(e_cv), 5)) + " at k: " + str(kmin[0])
    ax1[0][0].annotate(c_str, xy = (kmin[0]+0.1, min(e_cv)+0.01 ) )
    ax1[0][0].legend()
    ax1[0][0].set_title("2a. K Vs. Ecv")


    # ============================================================================================
    # for k = kmin[0], transform and calculate weights
    # then run sample error for train => e_samp

    centers, clusters = DataCluster(D, kmin[0])
    r = 1.0/np.sqrt(kmin[0])
    TD = RBF_Transform(D, centers, r)       # rbf transform data for kmin[0]
    Z = [td for td,y in TD]
    Y = [y for td,y in TD]
    W, temp = calcWeights(Z,Y)
    e_samp = SampError(TD, W)
    
    # ============================================================================================
    # NN decision boundary creation for k related to min(e_cv) for training data
    xlist = np.linspace(-1, 1, 200)
    ylist = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xlist, ylist)

    Z=[]
    for x_row, y_row in zip(X,Y):
        z_row=zip(x_row,y_row)
        Z.append(z_row)    
    k = kmin[0]
    r = 1.0/np.sqrt(kmin[0])
    nn=[]
    for z_row in Z:
        #transform z_row and predict
        row=[PredictY(pt, centers, r, W) for pt in z_row]
        nn.append(row)

    # ============================================================================================
    # plot decision boundary for kmin[0] related to min(e_cv) for training data
    # separate out the 1s and non-1s - training data
    I1, S1, Ix, Sx = separateOnes(train_d, train_I, train_S)
    cmap = colors.ListedColormap(['#dddddd', '#66ffff'])
    cp = ax1[1][0].contourf(X, Y, nn, cmap=cmap)
    ax1[1][0].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax1[1][0].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    fig.colorbar(cp, ax=ax1[1][0])
    ax1[1][0].set_xlabel("Total Intensity")
    ax1[1][0].set_ylabel("Vertical ASymmetry")
    ax1[1][0].legend()
    ax1[1][0].set_title("2b: Training Decision Boundary, k.min = " + str(k) + ": 1s and non-1s")


    # ============================================================================================
    # plot decision boundary for kmin[0] related to min(e_cv) for test data
    # separate out the 1s and non-1s - test data
    I1, S1, Ix, Sx = separateOnes(test_d, test_I, test_S)
    cmap = colors.ListedColormap(['#dddddd', '#66ffff'])
    cp = ax1[1][1].contourf(X, Y, nn, cmap=cmap)
    ax1[1][1].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax1[1][1].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    fig.colorbar(cp, ax=ax1[1][1])
    ax1[1][1].set_xlabel("Total Intensity")
    ax1[1][1].set_ylabel("Vertical ASymmetry")
    ax1[1][1].legend()
    ax1[1][1].set_title("2b: Test Decision Boundary, k.min = " + str(k) + ": 1s and non-1s")


    # ============================================================================================
    # now transform test data with kmin[0]
    # calculate e_test using weights W
    centers, clusters = DataCluster(Dtest, kmin[0])
    r = 1.0/np.sqrt(kmin[0])
    TD = RBF_Transform(Dtest, centers, r)       # rbf transform data for kmin[0]
    Z = [td for td,y in TD]
    Y = [y for td,y in TD]
    # W, temp = calcWeights(Z,Y)
    e_test = SampError(TD, W)

    print ("2a. Value of K chosen = ", kmin[0])
    print ("2b. E_cv = ", "{0:0.5f}".format(min(e_cv)) )
    print ("2b. E_in_sample = ", "{0:0.5f}".format(e_samp) )
    print ("2c. E_test = ", "{0:0.5f}".format(e_test) )

    plt.show()
"""
