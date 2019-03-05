import numpy as np
import matplotlib.pyplot as plt
import math

# To perform the training with a value of constant alpha, replace the "alpha" term in training function to "constantalpha"
constantalpha = 0.03
class neural:

    #Function defined to train the neural network (ADALINE)
    def training(self,epochs,trainingInput,trainingOutput):
        model = {}
        error = []
        weight = np.matrix('0 0;0 0;0 0')       #initial weight
        bias = np.matrix('0 0')     #initial bias
        alphas = []     #to store the decreasing alphas
        accuracies = []
        lms = []        #to store the least mean square errors
        for i in range(epochs):         #for loop which will run 'epochs' number of times
            alpha = 1 / (i + 1)         #alpha calculated by step function= 1/k where k is the cycle number
            alphas.append(alpha)

            for inp,output in zip(trainingInput,trainingOutput):
                modelOutput = inp*weight + bias
                absoluteError =  modelOutput- output
                error.append(absoluteError)
                weight = weight - 2*alpha*((inp.transpose())*absoluteError)
                bias = bias -2*alpha*absoluteError
                model['weight'] = weight
                model['bias'] = bias

            acc = self.accuracy(model, trainingInput, trainingOutput)
            accuracies.append(acc)
            epochError = self.meanSqueareError(model,trainingInput,trainingOutput)
            lms.append(epochError)
            #print("Accuracy at epoch : ", i, " is : ",acc , " and alpha = " , alpha)

        model['weight'] = weight
        model['bias'] = bias
        model['alphas'] = alphas
        model['accuracies'] = accuracies
        model['lms'] = lms
        return model

    #Function defined to calculate the mean square error(without activation function)
    def meanSqueareError(self,model,data,dataOutput):
        weight = model['weight']
        bias = model['bias']
        errorValue = 0
        for inp, output in zip(data,dataOutput):
            yin = inp*weight + bias
            error =yin - output
            error = np.square(error)        #Square the error
            errorValue = errorValue + np.sum(error)     #summation of squared errors
        return errorValue/11        #mean squared error

    #Function defined to calculate the Accuracy of the model
    def accuracy(self,model,data,dataOut):
        weight = model['weight']
        bias = model['bias']
        total = len(data)
        correct = 0
        for inp,output in zip(data,dataOut):
            modelOut = (inp*weight) + bias
            # Applying activation
            for i in range(np.size(output,1)):
                if(modelOut.item((0,i)) >= 0):
                    modelOut.itemset((0,i),1)
                else:
                    modelOut.itemset((0,i),-1)
            if ((modelOut - output) == 0).all():
                correct = correct+1
        accuracy = (correct/total)*100
        return accuracy

    #Function defined to plot the graphs
    def plotting(self,heading,xlabel,ylabel,x,y):
        plt.figure()
        plt.title(heading)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x,y, linewidth = 2.0)
        plt.grid(True)
        plt.show()

    #Function defined for the Logarithmic of errors to give better graph visualization
    def log10(self,elements):
        for i in range(len(elements)):
            elements[i] = math.log10(elements[i])
        return elements

    #Function to get the mean squared errors after applying activation function
    def lmsWithActivation(self,model,data,dataOutput):
        weight = model['weight']
        bias = model['bias']
        finalError = 0
        for inp,output in zip(data,dataOutput):
            y = inp*weight + bias
            for i in range(np.size(output,1)):
                if(y.item((0,i)) >= 0):
                    y.itemset((0,i),1)
                else:
                    y.itemset((0,i),-1)
            error = output - y
            error = np.square(error)
            finalError = finalError + np.sum(error)
        return finalError

#Function to display the weights, bias, accuracy at each epoch, alpha values, errors, error graph
def main():
    obj = neural()
    s1 = np.matrix('1 1 -1')
    s2 = np.matrix('1 2 -1')
    s3 = np.matrix('2 -1 1')
    s4 = np.matrix('2 0 1')
    s5 = np.matrix('1 -2 1')
    s6 = np.matrix('0 0 1')
    s7 = np.matrix('-1 2 1')
    s8 = np.matrix('-2 1 1')
    s9 = np.matrix('-1 -1 -1')
    s10 = np.matrix('-2 -2 -1')
    s11 = np.matrix('-2 -1 -1')
    o1 = np.matrix('-1 -1')
    o2 = np.matrix('-1 1')
    o3 = np.matrix('1 -1')
    o4 = np.matrix('1 1')
    trainData =[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11]     #training Data
    trainOutput = [o1,o1,o2,o2,o2,o2,o3,o3,o4,o4,o4]    #training output data
    epochs = 900
    model = obj.training(epochs,trainData,trainOutput)

    lmsWithActivation = obj.lmsWithActivation(model,trainData,trainOutput)
    print("The least mean square after applying activation is : ",lmsWithActivation)
    print("Weight : ",model['weight'])
    print("Bias : ", model['bias'])
    finalError = obj.meanSqueareError(model,trainData,trainOutput)
    print("Least Mean Square Error at the end : ", finalError)
    lms = model['lms']
    lms = obj.log10(lms)
    epochsArray =[]
    for i in range(1,901):
        epochsArray.append(i)

    alphas = model['alphas']
    obj.plotting("Logarithmic Mean SQ error vs epochs","epochs","log-error",epochsArray,lms)
    obj.plotting("Logarithmic Mean SQ error vs alphas","alpha","log-error", alphas[100:900],lms[100:900])

#Function to display the changes in the model after the changes in the values of x
def main2():
    neuralObject = neural()
    s1 = np.matrix('1 1 -1')
    s2 = np.matrix('1 2 -1')
    s3 = np.matrix('2 -1 1')
    s4 = np.matrix('2 0 1')
    s5 = np.matrix('1 -2 1')
    s6 = np.matrix('0 0 1')
    s7 = np.matrix('-1 2 1')
    s8 = np.matrix('-2 1 1')
    s9 = np.matrix('-1 -1 -1')
    s10 = np.matrix('-2 -2 -1')
    o1 = np.matrix('-1 -1')
    o2 = np.matrix('-1 1')
    o3 = np.matrix('1 -1')
    o4 = np.matrix('1 1')

    s11a = np.matrix('-2 -1 -1')        #changing values of x in [x -1 -1]
    s11b = np.matrix('1 -1 -1')
    s11c = np.matrix('4 -1 -1')
    s11d = np.matrix('7 -1 -1')
    s11e = np.matrix('11 -1 -1')
    s11f = np.matrix('12 -1 -1')


    train1a = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11a]       #creating training data
    train1b = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11b]
    train1c = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11c]
    train1d = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11d]
    train1e = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11e]
    train1f = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11f]

    trainOutput = [o1, o1, o2, o2, o2, o2, o3, o3, o4, o4, o4]
    epochs = 900

    modela = neuralObject.training(epochs, train1a, trainOutput)
    modelb = neuralObject.training(epochs, train1b, trainOutput)
    modelc = neuralObject.training(epochs, train1c, trainOutput)
    modeld = neuralObject.training(epochs, train1d, trainOutput)
    modele = neuralObject.training(epochs, train1e, trainOutput)
    modelf = neuralObject.training(epochs, train1f, trainOutput)

    errora = neuralObject.meanSqueareError(modela, train1a, trainOutput)
    errorb = neuralObject.meanSqueareError(modelb, train1b, trainOutput)
    errorc = neuralObject.meanSqueareError(modelc, train1c, trainOutput)
    errord = neuralObject.meanSqueareError(modeld, train1d, trainOutput)
    errore = neuralObject.meanSqueareError(modele, train1e, trainOutput)
    errorf = neuralObject.meanSqueareError(modelf, train1f, trainOutput)

    errors = [errora,errorb,errorc,errord,errore,errorf]
    print(errors)
    values = [-2,1,4,7,11,12]
    errorsInitital = [errora,errorb,errorc,errord,errore]
    valuesInitial = [-2,1,4,7,11]
    neuralObject.plotting("LMS Error vs x values", "x values", "LMS Error", valuesInitial, errorsInitital)
    neuralObject.plotting("LMS Error vs x values","x values","LMS Error",values,errors )

    accuracya = neuralObject.accuracy(modela, train1a, trainOutput)
    accuracyb = neuralObject.accuracy(modelb, train1b, trainOutput)
    accuracyc = neuralObject.accuracy(modelc, train1c, trainOutput)
    accuracyd = neuralObject.accuracy(modeld, train1d, trainOutput)
    accuracye = neuralObject.accuracy(modele, train1e, trainOutput)
    accuracyf = neuralObject.accuracy(modelf, train1f, trainOutput)
    accuracies1 = [accuracya,accuracyb,accuracyc,accuracyd,accuracye,accuracyf]
    print(accuracies1)
    neuralObject.plotting("Accuracy vs x Values", "x values", "Accuracy" , values,accuracies1)


if __name__ == '__main__':
    # to print the weights,bias, accuracies, alpha values and the error plotting for the original training set  =  uncomment line 35, uncomment line 219,comment line 220
    # to check the changes in the neural model after the changes in the value of x, comment line 35, comment line 219, uncomment line 220
    #main()
    main2()

