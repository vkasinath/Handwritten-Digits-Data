import numpy as np
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
        nm = np.matrix(arr[i])
        Iarr.append(np.sum(nm))

    return Iarr

def CalcVSymmetry(arr):
    Sarr = []
    for i in range(len(arr)):
        nm = np.matrix(arr[i])

        nmf = np.flipud(nm)         # flip the matrix up-down
        nps = np.absolute(np.subtract(nm, nmf))
        Sarr.append(np.sum(nps))

    return Sarr
    
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

    wlr = np.linalg.lstsq(X,Y, rcond=None)[0]
    m = -1.0*wlr[1]/wlr[2]
    c = -1.0*wlr[0]/wlr[2]


    # plot final images and graphs
    fig, (ax) = plt.subplots(2,2)
    plt.tight_layout()
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()

    # select random digit 1 and 5 (from train1 and train5), and plot gray image
    np.random.seed(1000)
    p1 = int(np.random.uniform(0,len(train1)))      # random digit 1
    ax[0][0].imshow(train1[p1], cmap='Greys')
    ax[0][0].set_title("Digit 1: Train sample #"+str(p1) )

    p1 = int(np.random.uniform(0,len(train5)))      # random digit 5
    ax[0][1].imshow(train5[p1], cmap='Greys')
    ax[0][1].set_title("Digit 5: Train sample# "+str(p1) )

    # scatter plot of features (intensity and Vertical ASymmetry) for training 1s and 5s
    ax[1][0].scatter(Itrain1, Strain1, marker='o', color="none", edgecolor='b', label='Train Digit 1')
    ax[1][0].scatter(Itrain5, Strain5, color='red', marker='x', label='Train Digit 5')

    # draw linear regression separation line for 1s and 5s feature scatter
    x = np.zeros(2); x[0] = -256.0; x[1] = 100.0
    y = x*m + (c-25)
    ax[1][0].plot(x, y, color='grey', linewidth=1.0)

    # pretty up the training data features graph
    ax[1][0].set_xlabel("Total Intensity")
    ax[1][0].set_ylabel("Vertical ASymmetry")
    ax[1][0].set_title("Feature Separation: Train Digits 1 & 5")
    ax[1][0].legend()

    # scatter plot of features (intensity and Vertical ASymmetry) for test 1s and 5s
    ax[1][1].scatter(Itest1, Stest1, marker='o', color="none", edgecolor='b', label='Test Digit 1')
    ax[1][1].scatter(Itest5, Stest5, color='red', marker='x', label='Test Digit 5')

    # draw linear regression separation line for test data, based on seperation line parameters obtained from train data
    x = np.zeros(2); x[0] = -256.0; x[1] = 100.0
    y = x*m + (c)
    ax[1][1].plot(x, y, color='grey', linewidth=1.0)

    # pretty up the test data features graph
    ax[1][1].set_xlabel("Total Intensity")
    ax[1][1].set_ylabel("Vertical ASymmetry")
    ax[1][1].set_title("Feature Separation: Test Digits 1 & 5")
    ax[1][1].legend()

    plt.show()
