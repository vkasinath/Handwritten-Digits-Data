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

# ===================================================================
def KNNSort(k, data, pt):
    dist_list = [(Distance(pt, x1x2),y) for x1x2,y in data]    
    k_neighbors = nsmallest(k, dist_list, key=lambda tup: tup[0])
    k_neighbors_y = [y for dist, y in k_neighbors]

    if np.sign(sum(k_neighbors_y)) > 0:
        return 1
    else:
        return -1

# ===================================================================
def Distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)


# ===================================================================
def KNN_crossvalidate(k, data):
    err = 0
    for pt in data:
        copy_d = copy.deepcopy(data)
        copy_d.remove(pt)        
        x1x2, y = pt
        if (y != KNNSort(k, copy_d, x1x2) ):
            err = err + 1
    e_cv = err/len(data)
    return e_cv

# ===================================================================
def KNN_samp_error(k, data):
    err = 0
    for pt in data:
        x1x2, y = pt
        if (y != KNNSort(k, data, x1x2) ):
            err = err + 1
    e_samp = err/len(data)
    return e_samp


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

    # create the training data points the way we need it for KNN [ ([x1, x2], Y) ]
    D = []
    for i in range(len(train_d)):
        D.append( ([train_I[i], train_S[i]], Y[i]) )

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
    # training data cross validation to determine k with least (e_cv)
    k_range = range(1,21)
    e_cv = []
    for k in k_range:
        e_cv.append( KNN_crossvalidate(k, D))

    # ============================================================================================
    # plot for k related to min(e_cv)
    # print("\n")
    # print("1a. Table of k, and Ecv")
    # print("Kvalue\t         Ecv")
    # print("------------\t------------")
    # for k in k_range:
    #    print("{0:0.5f}".format(k), "\t", "{0:0.5f}".format(e_cv[k-1]))

    kmin = [k_range[i-1] for i in k_range if e_cv[i-1] == min(e_cv)]

    # plot reg vs. Ecv (on training data)
    ax1[0][0].plot(k_range, e_cv, marker='.', color="k", label='Ecv')
    ax1[0][0].scatter(kmin[0], min(e_cv), marker='.', s=200, color="r", label='Min Ecv')
    ax1[0][0].set_xlabel("k values")
    ax1[0][0].set_ylabel("Ecv")

    c_str = "Min Ecv = " + str(np.around(min(e_cv), 5)) + " at k: " + str(kmin[0])
    ax1[0][0].annotate(c_str, xy = (kmin[0]+0.1, min(e_cv)+0.01 ) )
    ax1[0][0].legend()
    ax1[0][0].set_title("1b. K Vs. Ecv")


    # for k = kmin[0], run sample error for train = e_samp
    e_samp = KNN_samp_error(kmin[0], D)
    
    # for k = kmin[0], run sample error for test = e_test
    e_test = KNN_samp_error(kmin[0], Dtest)

    print ("1a. Value of K chosen = ", kmin[0])
    print ("1b. E_cv = ", "{0:0.5f}".format(min(e_cv)) )
    print ("1b. E_in_sample = ", "{0:0.5f}".format(e_samp) )
    print ("1c. E_test = ", "{0:0.5f}".format(e_test) )

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
    nn=[]
    for z_row in Z:
        row=[KNNSort(k, D, pt) for pt in z_row]
        nn.append(row)
    # ============================================================================================
    # separate out the 1s and non-1s - training data
    I1, S1, Ix, Sx = separateOnes(train_d, train_I, train_S)

    # plot for k related to min(e_cv) for training data
    cmap = colors.ListedColormap(['#dddddd', '#66ffff'])
    cp = ax1[1][0].contourf(X, Y, nn, cmap=cmap)
    ax1[1][0].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax1[1][0].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    fig.colorbar(cp, ax=ax1[1][0])
    ax1[1][0].set_xlabel("Total Intensity")
    ax1[1][0].set_ylabel("Vertical ASymmetry")
    ax1[1][0].legend()
    ax1[1][0].set_title("1b: Training Decision Boundary, k.min = " + str(k) + ": 1s and non-1s")

    # ============================================================================================
    # separate out the 1s and non-1s - test data
    I1, S1, Ix, Sx = separateOnes(test_d, test_I, test_S)

    # plot for k related to min(e_cv) for test data
    cmap = colors.ListedColormap(['#dddddd', '#66ffff'])
    cp = ax1[1][1].contourf(X, Y, nn, cmap=cmap)
    ax1[1][1].scatter(I1, S1, marker='o', color="none", edgecolor='b', label='Train:1')
    ax1[1][1].scatter(Ix, Sx, color='red', marker='x', label='Train non-1s')
    fig.colorbar(cp, ax=ax1[1][1])
    ax1[1][1].set_xlabel("Total Intensity")
    ax1[1][1].set_ylabel("Vertical ASymmetry")
    ax1[1][1].legend()
    ax1[1][1].set_title("1b: Test Decision Boundary, k.min = " + str(k) + ": 1s and non-1s")

    plt.show()
