import numpy as np
from operator import add
import matplotlib.pyplot as plt

def ParseData(filename):
    f = open(filename)
    i=0
    line = f.readline() 
    # save the data for 1s and 5s
    d1 = []
    d5 = []
    while(line != ""):
        i+=1
        fstr = str.split(line)
        num = int(float(fstr[0]))
        if (num == 1):
            img = []
            for i in range(16):
                row = []
                for j in range(16):
                    val = (i*16) + j
                    row.append(float(fstr[val+1]))
                img.append(row)
            d1.append(img)
#            print(num, img, "\n")
        elif (num == 5):
            img = []
            for i in range(16):
                row = []
                for j in range(16):
                    val = (i*16) + j
                    row.append(float(fstr[val+1]))
                img.append(row)
            d5.append(img)
#            print(num, img, "\n")

        line = f.readline() 
    f.close()
    return d1, d5
    

def CalcIntensity(arr):
    Iarr = []
    for i in range(len(arr)):
        Iarr.append(np.sum(arr[i]))

    return Iarr

def CalcVSymmetry(arr):
    Sarr = []
    for i in range(len(arr)):
        nmf = np.flipud(arr[i])         # flip the matrix up-down
        nps = np.absolute(np.subtract(arr[i], nmf))
        Sarr.append(np.sum(nps))

    return Sarr


def calcAccuracy(w, X, Y):
    n = Y.shape[0]
    predicted = np.empty(n)
    for i in range(n):      
        predY = np.dot(X[i],w)
        if (Y[i] == np.sign(predY)):
            predicted[i] = 1
        else:
            predicted[i] = 0
    return (np.sum(predicted)/n)
    

def calcEin(w, X, Y, etype):
    n = len(Y)

    wTx = np.dot(X,w)
    wTx_y = np.subtract(wTx,Y)
    enorm = np.linalg.norm(wTx_y)
    enorm_sq = enorm*enorm
    ein = enorm_sq/n
    if (etype == "train"):
        print("N=",n, "Enorm=",np.around(enorm,4), "Enorm_Sq=",np.around(enorm_sq,4), "Ein=",np.around(ein,4))
    if (etype == "test"):
        print("N=",n, "Enorm=",np.around(enorm,4), "Enorm_Sq=",np.around(enorm_sq,4), "Etest=",np.around(ein,4))
    
if __name__=="__main__":

    # parse and store digits 1 and 5 from train and test
    train1, train5 = ParseData("ZipDigits.train")
    test1, test5 = ParseData("ZipDigits.test")
    
    # features used = Intensity and Vertical ASymmetry
    # for train digit 1, calculate intensity and store it in Itrain1
    # for train digit 1, calculate Vertical ASymmetry and store it in Strain1
    Itrain1 = CalcIntensity(train1)
    Strain1 = CalcVSymmetry(train1)
 
    # for train digit 5, calculate intensity and store it in Itrain5
    # for train digit 5, calculate Vertical Symmetry and store it in Strain5
    Itrain5 = CalcIntensity(train5)
    Strain5 = CalcVSymmetry(train5)

    # for test digit 1, calculate intensity and store it in Itest1
    # for test digit 1, calculate Vertical ASymmetry and store it in Stest1
    Itest1 = CalcIntensity(test1)
    Stest1 = CalcVSymmetry(test1)
        
    # for test digit 5, calculate intensity and store it in Itest5
    # for test digit 5, calculate Vertical ASymmetry and store it in Stest5
    Itest5 = CalcIntensity(test5)
    Stest5 = CalcVSymmetry(test5)

    
    #3rd order transform of Itrain and Strain
    X = np.zeros((len(train1)+len(train5), 10))
    Y = np.zeros((len(train1)+len(train5), 1))
    for i in range(len(train1)):
        x1 = Itrain1[i]
        x2 = Strain1[i]
        x3 = Itrain1[i]**2
        x4 = Strain1[i]**2
        x5 = Itrain1[i]*Strain1[i]
        x6 = Itrain1[i]**3
        x7 = Strain1[i]**3
        x8 = Itrain1[i]*Itrain1[i]*Strain1[i]
        x9 = Itrain1[i]*Strain1[i]*Strain1[i]

        X[i] = np.asmatrix([1,x1,x2,x3,x4,x5,x6,x7,x8,x9])
        Y[i] = 1
                 
    for i in range(len(train5)):
        x1 = Itrain5[i]
        x2 = Strain5[i]
        x3 = Itrain5[i]**2
        x4 = Strain5[i]**2
        x5 = Itrain5[i]*Strain5[i]
        x6 = Itrain5[i]**3
        x7 = Strain5[i]**3
        x8 = Itrain5[i]*Itrain5[i]*Strain5[i]
        x9 = Itrain5[i]*Strain5[i]*Strain5[i]

        X[i+len(train1)] = np.asmatrix([1,x1,x2,x3,x4,x5,x6,x7,x8,x9])
        Y[i+len(train1)] = -1

    # multiple linear regression weights (pg 99)
    Xt = X.transpose()
    XtX = np.dot(Xt,X)
    XtXinv = np.linalg.inv(XtX)
    XtXinvXt = np.dot(XtXinv,Xt)
    wlr = np.dot(XtXinvXt,Y)

    print("3rd order Polynomial Transform weights =\n", np.around(np.squeeze(np.asarray(wlr)),4))

    accuracy = calcAccuracy(wlr,X,Y)
    calcEin(wlr,X,Y, "train")
    print("Train accuracy(%) = ", np.around(accuracy*100, 4))

    # plot final images and graphs
    fig, (ax) = plt.subplots(1,2)
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()

    # scatter plot of features (intensity and Vertical ASymmetry) for training 1s and 5s
    ax[0].scatter(Itrain1, Strain1, marker='o', color="none", edgecolor='b', label='Train Digit 1')
    ax[0].scatter(Itrain5, Strain5, color='red', marker='x', label='Train Digit 5')

    # draw contour/separation for train
    x = np.arange(-250,100, 1)
    y = np.arange(-250,100, 1)
    x1,y1 = np.meshgrid(x,y)
    z1 = wlr[0] + wlr[1]*x1 + wlr[2]*y1 + wlr[3]*(x1**2) + wlr[4]*(y1**2) + wlr[5]*x1*y1 + wlr[6]*(x1**3) + wlr[7]*(y1**3) + wlr[8]*(x1**2)*y1 + wlr[9]*x1*(y1**2)

    ax[0].contour(x1, y1, z1, [0])

    # pretty up the training data features graph
    ax[0].set_xlabel("Total Intensity")
    ax[0].set_ylabel("Vertical ASymmetry")
    ax[0].set_title("Feature Separation: Train Digits 1 & 5")
    ax[0].legend()

    # scatter plot of features (intensity and Vertical ASymmetry) for test 1s and 5s
    ax[1].scatter(Itest1, Stest1, marker='o', color="none", edgecolor='b', label='Test Digit 1')
    ax[1].scatter(Itest5, Stest5, color='red', marker='x', label='Test Digit 5')

    # test X, test Y to calculate test accuracy

    #3rd order transform of Itest and Stest
    X = np.zeros((len(test1)+len(test5), 10))
    Y = np.zeros((len(test1)+len(test5), 1))
    for i in range(len(test1)):
        x1 = Itest1[i]
        x2 = Stest1[i]
        x3 = Itest1[i]**2
        x4 = Stest1[i]**2
        x5 = Itest1[i]*Stest1[i]
        x6 = Itest1[i]**3
        x7 = Stest1[i]**3
        x8 = Itest1[i]*Itest1[i]*Stest1[i]
        x9 = Itest1[i]*Stest1[i]*Stest1[i]

        X[i] = np.asmatrix([1,x1,x2,x3,x4,x5,x6,x7,x8,x9])
        Y[i] = 1
                 
    for i in range(len(test5)):
        x1 = Itest5[i]
        x2 = Stest5[i]
        x3 = Itest5[i]**2
        x4 = Stest5[i]**2
        x5 = Itest5[i]*Stest5[i]
        x6 = Itest5[i]**3
        x7 = Stest5[i]**3
        x8 = Itest5[i]*Itest5[i]*Stest5[i]
        x9 = Itest5[i]*Stest5[i]*Stest5[i]

        X[i+len(test1)] = np.asmatrix([1,x1,x2,x3,x4,x5,x6,x7,x8,x9])
        Y[i+len(test1)] = -1

    accuracy = calcAccuracy(wlr,X,Y)
    calcEin(wlr,X,Y, "test")
    print("Test accuracy(%) = ", np.around(accuracy*100, 4))
    

    # draw contour/separation for test
    x = np.arange(-250,100, 1)
    y = np.arange(-250,100, 1)
    x1,y1 = np.meshgrid(x,y)
    z1 = wlr[0] + wlr[1]*x1 + wlr[2]*y1 + wlr[3]*(x1**2) + wlr[4]*(y1**2) + wlr[5]*x1*y1 + wlr[6]*(x1**3) + wlr[7]*(y1**3) + wlr[8]*(x1**2)*y1 + wlr[9]*x1*(y1**2)

    ax[1].contour(x1, y1, z1, [0])

    # pretty up the test data features graph
    ax[1].set_xlabel("Total Intensity")
    ax[1].set_ylabel("Vertical ASymmetry")
    ax[1].set_title("Feature Separation: Test Digits 1 & 5")
    ax[1].legend()

    plt.show()

