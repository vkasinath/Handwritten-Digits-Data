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

# ===================================================================
def DataCluster(ptsy,k):
    centers = []

    pts = [p for p,y in ptsy]
    # first random center
    rn = np.random.randint(0,len(pts))
    centers.append(pts[rn])
    
    # for next (n-1) centers 
    # for all points - calc euclidean distance for each point from centers
    # then get that one point, that is farthest from all centers

    for i in range(k-1):
        temp_dist = 0
        save_center = pts[0]
        for p in pts:
            dist_from_centers = [Distance(p, center) for center in centers]
            if (min(dist_from_centers) > temp_dist):
                save_center = p
                temp_dist = min(dist_from_centers)
        centers.append(save_center)

    clusters=[[center,[]] for center in centers]

    # 3 iterations
    for i in range(3):
        for p in pts:
            dist_from_centers=[Distance(p, center) for center in centers]
            clusters[np.argmin(dist_from_centers)][1].append(p)
        #calculate centroid
        centers=[]
        for [center,cluster_points] in clusters:
            x1_list = [x1 for [x1,x2] in cluster_points]
            x2_list = [x2 for [x1,x2] in cluster_points]
            center_x1=sum(x1_list)/len(x1_list)
            center_x2=sum(x2_list)/len(x2_list)
            centers.append([center_x1, center_x2])

    return centers, clusters

# ===================================================================
def RBF_Transform(data, centers, r):
    TD = []
    for x1x2,y in data:
        trans_x1x2 = [1] + [np.e**(-0.5*(Distance(center, x1x2)/r)**2) for center in centers]
        TD.append( (trans_x1x2,y) )

    return TD

# ===================================================================
def calcWeights(Z,Y):

    L = 0
    numx = len(Z[0])
    ZT = np.transpose(Z)
    ZTZ = np.dot(ZT,Z)

    I = np.identity(numx)
    ZTZ_LI = np.add(ZTZ, L*I)
    
    ZTZ_LI_inv = np.linalg.inv(ZTZ_LI)
    ZTZ_LI_inv_ZT = np.dot(ZTZ_LI_inv, ZT)
    W = np.dot(ZTZ_LI_inv_ZT, Y)

    return W, ZTZ_LI_inv_ZT


# ===================================================================
# leave-one-out-cross validation - calculate Ecv
def LOOV(TD):

    err = 0.0
    Z = [td for td,y in TD]
    Y = [y for td,y in TD]

    for n in range(len(Z)):
        # Z1 = Z with nth record out. similarly Y1 = Y with nth record out.
        Z1 = np.delete(Z, n, 0)
        Y1 = np.delete(Y, n, 0)

        # calculate regression weights W and H
        W, ZTZ_LI_inv_ZT = calcWeights(Z1,Y1)

        # using W, predict for the nth record            
        Y_pred = np.dot(np.transpose(W), Z[n])

        # keep summing up if any error on the 1 record held for validation
        if(np.sign(Y_pred) != np.sign(Y[n])):
            err = err + 1

    # Ecv across all the N records (formula above fig 4.13)
    e_cv = err/len(Z)

    return e_cv

# ===================================================================
# calculate samp error
def SampError(TD, W):

    err = 0.0
    Z = [td for td,y in TD]
    Y = [y for td,y in TD]

    for n in range(len(Z)):

        # using W, predict for the nth record            
        Y_pred = np.dot(np.transpose(W), Z[n])

        # keep summing up if any error on the 1 record held for validation
        if(np.sign(Y_pred) != np.sign(Y[n])):
            err = err + 1

    # Ecv across all the N records (formula above fig 4.13)
    e_samp = err/len(Z)

    return e_samp


# ===================================================================
# predictY - for a given point - calculate the transformed x-vals to predict Y classification
def PredictY(pt, centers, r, W):
    Xs = [1] + [np.e**(-0.5*(Distance(center, pt)/r)**2) for center in centers]

    # using W, predict for the nth record            
    Y_pred = np.dot(np.transpose(W), Xs)

    return np.sign(Y_pred[0])



if __name__=="__main__":

    np.random.seed(75)

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
        # k is the number of centers. create the centers, and the
        # cluster of datapoints around k-centers
        # then RBF-Transform the data
        # then calculate weights
        # then cross-validate on training data
        # store E_cv for each k, so we can choose k with min(E_cv)

        centers, clusters = DataCluster(D, k)
        r = 1.0/np.sqrt(k)
        TD = RBF_Transform(D, centers, r)       # rbf transform data
        cv_error = LOOV(TD)
        e_cv.append(cv_error)


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
    W, temp = calcWeights(Z,Y)
    e_test = SampError(TD, W)

    print ("2a. Value of K chosen = ", kmin[0])
    print ("2b. E_cv = ", "{0:0.5f}".format(min(e_cv)) )
    print ("2b. E_in_sample = ", "{0:0.5f}".format(e_samp) )
    print ("2c. E_test = ", "{0:0.5f}".format(e_test) )

    plt.show()
