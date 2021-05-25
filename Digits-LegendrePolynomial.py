import numpy as np
from operator import add
import matplotlib.pyplot as plt
import random


LO = 8                      # 8th order legendre polynomial transform

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
# Legendre Polynomial upto order 8
def LPoly(ord, x):
    if   (ord == 0):      return 1
    elif (ord == 1):      return x
    elif (ord == 2):      return (-1 + 3*x**2)/2.0
    elif (ord == 3):      return (-3*x + 5*x**3)/2.0
    elif (ord == 4):      return (3 - 30*x**2 + 35*x**4)/8.0
    elif (ord == 5):      return (15*x - 70*x**2 + 63*x**5)/8.0
    elif (ord == 6):      return (-5 + 105*x**2 - 315*x**4 + 231*x**6)/16.0
    elif (ord == 7):      return (-35*x + 315*x**3 - 693*x**5 + 429*x**7)/16.0
    elif (ord == 8):      return (35 - 1260*x**2 + 6930*x**4 - 12012*x**6 + 6435*x**8)/128.0
    else:                 return 0

# ===================================================================
# returns an array of the legendre poly transformation of x1 and x2
def LTrans(x1, x2):
    X = [LPoly(i,x1)*LPoly((ord-i),x2) for ord in range(LO+1) for i in range(ord+1)]
    return X

# ===================================================================
# calculate weights for transformed features
def calcWeights(Z,L):

    numx = len(Z[0])
    ZT = np.transpose(Z)
    ZTZ = np.dot(ZT,Z)

    I = np.identity(numx)
    ZTZ_LI = np.add(ZTZ, L*I)
    
    ZTZ_LI_inv = np.linalg.inv(ZTZ_LI)
    ZTZ_LI_inv_ZT = np.dot(ZTZ_LI_inv, ZT)
    W = np.dot(ZTZ_LI_inv_ZT,Y)

    return W, ZTZ_LI_inv_ZT

# ===================================================================
# leave-one-out-validation - calculate Ecv and Etest
def LOOV(reg, Z, Y, Zt, Yt):

    # for each reg(lambda), calculate Ecv using equatino (4.13)
    # store Ecv for each lambda

    Ecv = []
    Etest = []

    for L in reg:
        # calculate regression weights W and H for each L (regularization Lambda)
        W, ZTZ_LI_inv_ZT = calcWeights(Z,L)
        H = np.dot(Z, ZTZ_LI_inv_ZT)
        Y_bar = np.dot(H,Y)

        # calculate Ecv (analytic formula)
        err = 0.0
        for n in range(len(Z)):
            numer = (Y_bar[n] - Y[n])
            denom = (1-H[n][n])
            val = (numer/denom)**2
            err += val
        Ecv.append(err.item()/len(Z))


        # use the weights W, predict Y and calculate Etest (from actual Y in test data)
        # calculate Etest
        err = 0.0
        for n in range(len(Zt)):
            Y_pred = np.dot(np.transpose(W), Zt[n])
            if(np.sign(Y_pred) != np.sign(Yt[n])):
                err = err + 1

        Etest.append(err/len(Zt))

    return Ecv, Etest

# ===================================================================
# separate the 1s and non-1s (for scatter plots)
def separateOnes(train_d, train_I, train_S):
    I1 = [train_I[i] for i in range(len(train_d)) if train_d[i] == 1]
    S1 = [train_S[i] for i in range(len(train_d)) if train_d[i] == 1]

    Ix = [train_I[i] for i in range(len(train_d)) if train_d[i] != 1]
    Sx = [train_S[i] for i in range(len(train_d)) if train_d[i] != 1]
    return I1, S1, Ix, Sx



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
    random.seed(100)
    N = len(digit)
    R = random.sample(range(0, len(digit)), 300)

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

    # ===================================================================
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

    # ===================================================================
    # Q1 find out how many x values for a legendre poly tranform
    numx = 0
    for ord in range(LO+1):
        for i in range(ord+1):
            j = ord - i
            numx = numx + 1
            print("[" + str(j) + "," + str(i) + "] ", end='')
        print("")

    print("\nQ1: Dimensions of Train Z: Rows x Cols =", len(train_d), "x", numx)

    # ============================================================================================
    # create Z = 8th order Legendre Poly transform I_n, S_n for train data
    # create Zt = 8th order Legendre Poly transform I_n, S_n for test data
    Z = [LTrans(train_I[row], train_S[row]) for row in range(len(train_d))]
    Zt = [LTrans(test_I[row], test_S[row]) for row in range(len(test_d))]

    # ============================================================================================
    # create framework for graphs and plots [3 graphs with subplots inside]. for Q2 and Q4
    fig, (ax) = plt.subplots(1, 2)
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()

    # create framework for graphs and plots. for Q4
    fig, (ax1) = plt.subplots(1, 1)
    plt.tight_layout()
    ax1.grid()
    #ax1[1].grid()

    # create framework for graphs and plots. for Q5
    fig, (ax2) = plt.subplots(1, 2)
    plt.tight_layout()
    ax2[0].grid()
    ax2[1].grid()

    # ============================================================================================
    # Q2
    # calculate regression weights without regularization (lambda = 0)
    W, temp = calcWeights(Z, 0.0)

    # Q2 plot contour/decision boundary of training data (separate blues and reds 1s and non-1s)
    # will be used for Q3 and first part of Q5
    I1, S1, Ix, Sx = separateOnes(train_d, train_I, train_S)

    ax[0].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax[0].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    ax[0].set_xlabel("Total Intensity")
    ax[0].set_ylabel("Vertical ASymmetry")
    ax[0].legend()
    ax[0].set_title("Q2: Feature Separation Train Lambda = 0.0: 1s and non-1s")

    # Q2 draw contour/separation in train AND test graphs
    mx = np.arange(-1.2,1.2, 0.05)
    my = np.arange(-1.2,1.2, 0.05)
    m1, m2 = np.meshgrid(mx,my)

    ltz=[LTrans(x1, x2) for x1,x2 in zip(m1,m2)]
    z = [sum([coef*feat for coef,feat in zip(W,t)]) for t in ltz]
    ax[0].contour(m1, m2, z, [0])

    # ============================================================================================
    # Q3
    # calculate regression weights without regularization (lambda = 2.0)
    W, temp = calcWeights(Z, 2.0)

    ax[1].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax[1].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    ax[1].set_xlabel("Total Intensity")
    ax[1].set_ylabel("Vertical ASymmetry")
    ax[1].legend()
    ax[1].set_title("Q3: Feature Separation Train Lambda = 2.0: 1s and non-1s")

    # Q3 draw contour/separation in train AND test graphs
    mx = np.arange(-1.2,1.2, 0.05)
    my = np.arange(-1.2,1.2, 0.05)
    m1, m2 = np.meshgrid(mx,my)

    ltz=[LTrans(x1, x2) for x1,x2 in zip(m1,m2)]
    z = [sum([coef*feat for coef,feat in zip(W,t)]) for t in ltz]
    ax[1].contour(m1, m2, z, [0])

    # ============================================================================================
    # Q4: use Leave-one-out-validation to calculate Ecv on the training set, for each lambda
    # Q4: calculate Etest on the test set, for each lambda
    reg = [0, 0.00001, 0.0001, 0.001,  0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2, 2.2, 2.4, 2.6, 2.8, 3.0]
    Ecv, Etest = LOOV(reg, Z, Y, Zt, Yt)

    print("\n")
    print("Q4. Table of Lambda, Ecv and Etest")
    print("Lambda\t         Ecv  \t         Etest ")
    print("------------\t------------\t------------")
    for i in range(len(reg)):
        print("{0:0.5f}".format(reg[i]), "\t", "{0:0.5f}".format(Ecv[i]), "\t", "{0:0.5f}".format(Etest[i]))

    # plot reg vs. Ecv (on training data)
    ax1.plot(reg, Ecv, marker='.', color="r", label='Ecv')
    ax1.plot(reg, Etest, marker='.', color="b", label='Etest')
    ax1.set_xlabel("Lambda, Regularization")
    ax1.set_ylabel("Error (cv, test)")

    c_str = "Min Ecv = " + str(np.around(min(Ecv), 5)) + " at Lambda: " + str([reg[i] for i in range(len(reg)) if Ecv[i] == min(Ecv)])
    ax1.annotate(c_str, xy = (0.25, 0.06))
    ax1.legend()
    ax1.set_title("Q4. Lambda Vs. Ecv, Etest")

    # ============================================================================================
    # Q5 calculate regression weights without regularization (lambda for min(Ecv) )
    lmin = [reg[i] for i in range(len(reg)) if Ecv[i] == min(Ecv)]
    W, temp = calcWeights(Z, lmin)

    ax2[0].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax2[0].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    ax2[0].set_xlabel("Total Intensity")
    ax2[0].set_ylabel("Vertical ASymmetry")
    ax2[0].legend()
    ax2[0].set_title("Q5: Feature Separation Train Lambda = " + str(lmin) + ": 1s and non-1s")

    # Q5 draw contour/separation in train AND test graphs
    mx = np.arange(-1.2,1.2, 0.05)
    my = np.arange(-1.2,1.2, 0.05)
    m1, m2 = np.meshgrid(mx,my)

    ltz=[LTrans(x1, x2) for x1,x2 in zip(m1,m2)]
    z = [sum([coef*feat for coef,feat in zip(W,t)]) for t in ltz]
    ax2[0].contour(m1, m2, z, [0])

    # plot contour/decision boundary of test data
    # Q5 plot contour/decision boundary of test data (separate blues and reds, 1s and non-1s)
    I1, S1, Ix, Sx = separateOnes(test_d, test_I, test_S)

    ax2[1].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Test:1')
    ax2[1].scatter(Ix, Sx, color='red', marker='x', label='Test non-1s')
    ax2[1].set_xlabel("Total Intensity")
    ax2[1].set_ylabel("Vertical ASymmetry")
    ax2[1].legend()
    ax2[1].set_title("Q5: Feature Separation Test Lambda = " + str(lmin) + ": 1s and non-1s")

    # draw contour/separation in train AND test graphs
    mx = np.arange(-1.2,1.2, 0.05)
    my = np.arange(-1.2,1.2, 0.05)
    m1, m2 = np.meshgrid(mx,my)

    ltz=[LTrans(x1, x2) for x1,x2 in zip(m1,m2)]
    z = [sum([coef*feat for coef,feat in zip(W,t)]) for t in ltz]
    ax2[1].contour(m1, m2, z, [0])

    plt.show()
