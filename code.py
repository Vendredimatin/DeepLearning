import numpy as np
import time

## Network architecture
NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons

## Hyperparameters
NUM_HIDDEN = 50
LEARNING_RATE = 0.03
BATCH_SIZE = 64
NUM_EPOCH = 300

print("NUM_HIDDEN: ", NUM_HIDDEN)
print("LEARNING_RATE: ", LEARNING_RATE)
print("BATCH_SIZE: ", BATCH_SIZE)
print("NUM_EPOCH: ", NUM_EPOCH)


class ForwardPropagationRet:
    Z1 = 0
    H1 = 0
    Z2 = 0
    Y_hat = 0


class Delta:
    deltaW2 = 0
    deltab2 = 0
    deltaW1 = 0
    deltab1 = 0


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("./data/mnist_{}_images.npy".format(which))
    labels = np.load("./data/mnist_{}_labels.npy".format(which))
    return images, labels


## 1. Forward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute the cross-entropy (CE) loss.
def fCE(X, Y, W1, b1, W2, b2):
    # print(X.shape)
    ## your code here
    #X = np.array(X).reshape(NUM_INPUT, 1)
    # X = 64 * 784 Y = 64 * 10 b1 = 1*50 b2= 1*10 W1 = 784 * 50 W2 = 50 * 10
    Z1 = np.dot(X, W1) + b1  # 64 * 50
    H1 = ReLU(Z1) # 64 * 50
    Z2 = np.dot(H1,W2) + b2 # 64 * 10
    Y_hat = Softmax(Z2) # 64 * 10

    loss = LOSS(BATCH_SIZE, Y, Y_hat)
    ret = ForwardPropagationRet()
    ret.Y_hat = Y_hat
    ret.H1 = H1
    ret.Z2 = Z2
    ret.Z1 = Z1

    return loss, ret


## 2. Backward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute the gradient of fCE.
def gradCE(X, Y, W1, b1, W2, b2, fpRes):
    ## your code here
    
    Y_hat = fpRes.Y_hat
    H1 = fpRes.H1
    Z1 = fpRes.Z1

    a = np.sign(Z1)
    b = np.int64(Z1 > 0)
    
    delta_y = Y_hat - Y
    deltaW2 = H1.T.dot(delta_y)/BATCH_SIZE # 50*64 64*10 = 50*10
    deltab2 = np.sum(delta_y,axis=0)/BATCH_SIZE   #  1 * 10
    deltaW1 = X.T.dot(delta_y.dot(W2.T) * np.sign(Z1))/BATCH_SIZE # 784 * 50
    deltab1 = np.sum(delta_y.dot(W2.T) * np.sign(Z1), axis = 0)/BATCH_SIZE # 1 * 50

    # for i in range(BATCH_SIZE):
    #     deltaW2 = deltaW2 + H1[i].T * (Y_hat[i] - Y[i]) # 50 * 10
    #     deltab2 = deltab2 + Y_hat[i] - Y[i]
    #     deltaW1 = deltaW1 + W2.T * (Y_hat[i] - Y[i]) * np.sign(Z1[i]) * X[i].T # 784 * 50
    #     deltab1 = deltab1 + W2.T * (Y_hat[i] - Y[i]) * np.sign(Z1[i]) # 1 * 50

    # axis = 0 表示列， axis = 1 表示行
    delta = Delta()
    delta.deltaW2 = deltaW2
    delta.deltaW1 = deltaW1
    delta.deltab2 = deltab2
    delta.deltab1 = deltab1

    return delta


## 3. Parameter Update
# Given training and testing datasets, train the NN.
def train(trainX, trainY, testX, testY):
    #  Initialize weights randomly
    W1 = np.random.randn(NUM_INPUT, NUM_HIDDEN)
    W2 = np.random.randn(NUM_HIDDEN, NUM_OUTPUT)
    b1 = np.random.randn(1,NUM_HIDDEN)
    b2 = np.random.rand(1, NUM_OUTPUT)
    N = len(trainX)
    iter = int((N / BATCH_SIZE))

    for i in range(NUM_EPOCH):
        # 每次epoch之前打乱数据集
        index = np.arange(N)
        np.random.shuffle(index)
        trainX = trainX[index]
        trainY = trainY[index]
        total_loss = 0
        for j in range(iter + 1):
            start = j * BATCH_SIZE
            X = trainX[start:start + BATCH_SIZE]
            Y = trainY[start:start + BATCH_SIZE]
            # X = np.reshape()
            loss, fpRes = fCE(X, Y, W1, b1, W2, b2)
            total_loss += loss
            delta = gradCE(X, Y, W1, b1, W2, b2, fpRes)

            # update parameters
            W2 = W2 - LEARNING_RATE * delta.deltaW2
            W1 = W1 - LEARNING_RATE * delta.deltaW1
            b1 = b1 - LEARNING_RATE * delta.deltab1
            b2 = b2 - LEARNING_RATE * delta.deltab2

        # 在训练集上测试结果
        print("epoch:(%d), batch:(%d), loss:(%.5f)"%(i,j,total_loss/iter))
        train_loss, train_fp_res = fCE(trainX, trainY, W1, b1, W2, b2)
        train_y_hat = train_fp_res.Y_hat
        train_predict_label = np.argmax(train_y_hat, axis=1)
        train_truth_label = np.argmax(trainY, axis=1)

        train_accuracy = np.sum(train_predict_label == train_truth_label)/len(trainY)
        print("epoch:(%d),train_loss:(%.5f) train_accuracy:(%.5f)--------------" %(i,train_loss,train_accuracy))


        # 在测试集上测试结果
        test_loss, test_fp_res = fCE(testX, testY, W1, b1, W2, b2)
        test_y_hat = test_fp_res.Y_hat
        test_predict_label = np.argmax(test_y_hat, axis=1)
        test_truth_label = np.argmax(testY, axis=1)

        test_accuracy = np.sum(test_predict_label == test_truth_label)/len(testY)
        print("epoch:(%d),test_loss:(%.5f) test_accuracy:(%.5f)--------------" %(i,test_loss,test_accuracy))

       
    ## your code here

    print("completed!")
    pass


def ReLU(X):
    return np.maximum(X, 0)


def Softmax(X):
    # dim(X) = n * 10
    # 计算每行的最大值
    X_row_max = X.max(axis=1)
 
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    X_row_max = X_row_max.reshape(-1, 1)
    X = X - X_row_max
 
    # 计算e的指数次幂
    X_exp = np.exp(X)
    #因为每个batch中有n个，对batch中每组（即行）进行求和
    X_sum = np.sum(X_exp, axis=1, keepdims=True)

    return X_exp/X_sum


def LOSS(n, Y, Y_hat):
    loss = -1/n * np.sum(Y * np.log(Y_hat))
    return loss


if __name__ == "__main__":
    # Load data
    start_time = time.time()
    trainX, trainY = loadData("train")
    testX, testY = loadData("test")

    print("len(trainX): ", len(trainX))
    print("len(testX): ", len(testX))

    # # Train the network and report the accuracy on the training and test set.
    train(trainX, trainY, testX, testY)