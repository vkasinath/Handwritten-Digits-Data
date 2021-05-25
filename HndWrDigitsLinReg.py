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
    n = len(Y)
    predicted = np.empty(n)

    for i in range(n):
        predY = np.sign(np.inner(w,X[i]))

        if (Y[i] == predY):
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

    
    # calculate linear regression; separation line for train1 and train5
    X = []
    Y = []
    for i in range(len(train1)):
        X.append([1,Itrain1[i],Strain1[i]])
        Y.append(1)
                 
    for i in range(len(train5)):
        X.append([1,Itrain5[i],Strain5[i]])
        Y.append(-1)

    # linear regression weights
    wlr = np.linalg.lstsq(X,Y, rcond=None)[0]
    accuracy = calcAccuracy(wlr,X,Y)

    m = -1.0*wlr[1]/wlr[2]
    c = -1.0*wlr[0]/wlr[2]

    print("final weights = ", np.around(wlr,4))

    # plot final images and graphs
    fig, (ax) = plt.subplots(1,2)
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()

    # scatter plot of features (intensity and Vertical ASymmetry) for training 1s and 5s
    ax[0].scatter(Itrain1, Strain1, marker='o', color="none", edgecolor='b', label='Train Digit 1')
    ax[0].scatter(Itrain5, Strain5, color='red', marker='x', label='Train Digit 5')
    accuracy = calcAccuracy(wlr,X,Y)
    calcEin(wlr,X,Y, "train")
    print("Training accuracy(%) = ", np.around(accuracy*100, 4))

    # Obtain linear regression weights for 1 & 5 feature separation
    x = np.zeros(2); x[0] = -256.0; x[1] = 100.0
    y = x*m + (c-25)
    ax[0].plot(x, y, color='grey', linewidth=1.0)


    # pretty up the training data features graph
    ax[0].set_xlabel("Total Intensity")
    ax[0].set_ylabel("Vertical ASymmetry")
    ax[0].set_title("Feature Separation: Train Digits 1 & 5")
    ax[0].legend()

    # scatter plot of features (intensity and Vertical ASymmetry) for test 1s and 5s
    ax[1].scatter(Itest1, Stest1, marker='o', color="none", edgecolor='b', label='Test Digit 1')
    ax[1].scatter(Itest5, Stest5, color='red', marker='x', label='Test Digit 5')

    # test X, test Y to calculate test accuracy
    X = []
    Y = []
    for i in range(len(test1)):
        X.append([1,Itest1[i],Stest1[i]])
        Y.append(1)
                 
    for i in range(len(test5)):
        X.append([1,Itest5[i],Stest5[i]])
        Y.append(-1)
    accuracy = calcAccuracy(wlr,X,Y)
    calcEin(wlr,X,Y, "test")
    print("Test accuracy(%) = ", np.around(accuracy*100, 4))

    # draw linear regression separation line for test data, based on seperation line parameters obtained from train data
    x = np.zeros(2); x[0] = -256.0; x[1] = 100.0
    y = x*m + (c)
    ax[1].plot(x, y, color='grey', linewidth=1.0)

    # pretty up the test data features graph
    ax[1].set_xlabel("Total Intensity")
    ax[1].set_ylabel("Vertical ASymmetry")
    ax[1].set_title("Feature Separation: Test Digits 1 & 5")
    ax[1].legend()

    plt.show()
