import random
import math
# import numpy as np

class annImage:
    nodeHidden = 100
    momentum=0.3
    learning=0.05/752
    nodeinput =4200
    jumTrain=0
    jumTest=0
    errorMax = 0.001
    epoch = 500
    trainLabel =[]
    train = []
    testLabel = []
    test = []
    wInput =[]
    wHidden1 = []
    wHidden2 = []
    hw1 = []
    hw2 = []
    h1 = []
    h2 = []
    o = []
    outputNet = []
    errorO=0
    outputUnit = []
    hUnit1 = []
    hUnit2 = []
    wInL = []
    wHid1L = []
    wHid2L = []
    biasHidden1 = []
    biasHidden2 = []
    biasO = 0

    def loadData(arg,table,length,file):#memasukkan data label pada list
        f = open(file, "r")
        b=0
        for line in f:
            a=line.strip().split("\n")
            if a[0] =='1':
                b=1
            elif a[0] == '0':
                b=0
            table.append(b)
            length += 1
        return length



    def load(arg, table, file):#mengubah dataset ke dalam biner dan disimpan dalam list
        f = open(file, "r")
        biner = []
        b=0
        for line in f:
            tuple=line
            b+=1
            for i in tuple:
                if i =="#":
                    biner.append(1)
                elif i == "\n":
                    continue
                else:
                    biner.append(0)
            if b % 70 == 0 :
                table.append(biner)
                biner = []


    def bobotAwal(arg, input, node,weighth,weighth1,weighth2,weighthU,weighth1U,weighth2U,biasHidden1,biasHidden2,biasO):  # inisialisasi bobot Awal
        bound=1/math.sqrt(input)
        weight =[]
        wu=[]
        a= 0
        random.seed(0)
        for i in range (node):#perulangan sebanyak node hidden layer
            for j in range (input):#perulangan sebanyak node input
                 a= random.uniform(-bound, bound)
                 weight.append(a)
                 wu.append(0)
            weighth.append(weight)
            weighthU.append(wu)
            weight=[]
            wu = []
            biasHidden1.append(random.uniform(-bound, bound))
            biasHidden2.append(random.uniform(-bound, bound))

        # print (weighth)
        # print(str(len(weighth)))

        random.seed(0)
        for k in range(node):#perulangan sebanyak node hidden layer1
            for l in range (node):#perulangan sebanyk node hidden layer2
                 a= random.uniform(-bound, bound)
                 weight.append(a)
                 wu.append(0)
            weighth1.append(weight)
            weighth1U.append(wu)
            weight=[]
            wu = []

        # print (weighth1)
        # print(str(len(weighth1)))

        random.seed(0)
        for m in range (node):  # perulangan sebanyak node hidden layer2
            a = random.uniform(-bound, bound)
            weighth2.append(a)
            weighth2U.append(0)
        biasO=random.uniform(-bound, bound)

        # print(str(len(weighth2)))
        return biasO


    def hitungH1(arg,weighth, node, input, hnode,honode, fitur, biasHidden1,update):  # menghitung nilai h
        temphnode=0
        ho=0
        sigma=[]
        hidden =[]
        print(str(len(hnode)))
        print(str(len(input)))
        for k in range(len(input)):  # perulangan sebanyak jumlsh data input
            for i in range(node):#perulangan sebanyak node hidden layer
                for j in range (fitur):#perulangan sebanyak node input
                    temphnode = temphnode+ weighth[i][j]*input[k][j]
                temphnode = temphnode + (1 * biasHidden1[i])
                ho= 1 / (1 + (math.exp(-(temphnode))))
                if(update==False):
                    sigma.append(temphnode)
                    hidden.append(ho)
                elif (update == True):
                    hnode[k][i] = temphnode
                    honode[k][i] = ho
                temphnode=0
            if (update == False):
                hnode.append(sigma)
                honode.append(hidden)
                sigma = []
                hidden = []

        # print(hnode)
        # print(honode)

        # print("h1------------------------------------")
        # print(str(len(hnode)))
        # print(str(len(honode)))
        #
        # print(weighth[0][1])
        # print(weighth[1][1])


    def hitungH2(arg,weighth, node, h, hnode,honode,biasHidden2,update):  # menghitung nilai h
        temphnode=0
        ho=0
        sigma = []
        hidden = []
        # print (str(input[0]))
        for k in range(len(h)):  # perulangan sebanyak jumlsh data input
            for i in range(node):#perulangan sebanyak node hidden layer
                for j in range(node):#perulangan sebanyak node hidden 1
                    temphnode = temphnode + weighth[i][j] * h[k][j]
                temphnode = temphnode + (1 * biasHidden2[i])
                ho = 1 / (1 + (math.exp(-(temphnode))))
                if (update == False):
                    sigma.append(temphnode)
                    hidden.append(ho)
                elif (update == True):
                    hnode[k][i] = temphnode
                    honode[k][i] = ho
                temphnode=0
            if (update == False):
                hnode.append(sigma)
                honode.append(hidden)
        # print("h2------------------------------------")
        # print(str(len(hnode)))
        # print(str(len(honode)))


    def hitungOutput(arg,weighth, node, h, outputN,nilaiO,biasO,update):  # menghitung nilai output
        temponode=0
        o=0
        for k in range(len(h)):  # perulangan sebanyak jumlsh data input
            for i in range(node):#perulangan sebanyak node hidden layer2
                temponode = temponode + weighth[i] * biasO
            temponode = temponode + 1 * biasO
            o = 1 / (1 + (math.exp(-(temponode))))
            if (update == False):
                outputN.append(temponode)
                nilaiO.append(o)
            elif (update == True):
                outputN[k] = temponode
                nilaiO[k] = o
            temponode=0
        # print("OUT------------------------------------")
        # print(str(len(outputN)))
        # print(str(len(nilaiO)))


    def error(arg,target, net,totalError):  # menghitung nilai h
        error=0
        totalError=0
        print(len(target))
        print(len(net))
        for i in range(len(net)):  # perulangan sebanyak jumlsh data input
            error = error + 0.5 *(math.pow((target[i]-net[i]),2))
        totalError = error
        # print("err-----------------------------")
        # print(str(totalError))
        return totalError

    def deltaOutput(arg, target, net, outputUnit,update):  # menghitung nilai h
        for i in range(len(net)):
            deltaO = net[i] *(1 - net[i]) * (target[i] - net[i])
            if (update == False):
                outputUnit.append(deltaO)
            elif (update == True):
                outputUnit[i] = deltaO
        # print("Ounit-----------------------------")
        # print(len(outputUnit))

    def deltaHidden2(arg, neth2, node,  outputU, weighth2, hU2,update ):  # menghitung nilai h
        temphdelta=0
        delt=[]
        # print(str(neth2[0]))
        for i in range(len(outputU)):
            for j in range(node):#perulangan sebanyak node hidden layer2
                temphdelta= neth2[i][j] * ( 1 - neth2[i][j]) * (outputU[i] * weighth2[j])
                if (update == False):
                    delt.append(temphdelta)
                elif (update == True):
                    hU2[i][j]= temphdelta
            if (update == False):
                hU2.append(delt)
                delt=[]
            temphdelta = 0
        # print("H2unit-----------------------------")
        # print(len(hU2))

    def deltaHidden1(arg, neth1, node, hU2, weighth1, hU1,update):  # menghitung nilai h
        temphdelta = 0
        h1=0
        delt = []
        for k in range(len(hU2)):
            for i in range(node):  # perulangan sebanyak node hidden layer1
                for j in range(node): # perulangan sebanyak node hidden layer2
                    temphdelta = temphdelta +  (hU2[k][j] * weighth1[j][i])
                h1= neth1[k][i] * (1 - neth1[k][i]) * temphdelta
                if (update == False):
                    delt.append(h1)# Net
                elif (update == True):
                    hU1[k][i] = h1
            if (update == False):
                hU1.append(delt)
                delt = []
            temphdelta = 0
        # print("H1unit-----------------------------")
        # print(len(hU1))


    def deltaWHidden2(arg, node, weighthL,weighth,momentum,learn, outputU, neth2):  # menghitung perubahan weight
        deltao=0;
        weight = 0;
        for i in range (node):  # perulangan sebanyak node hidden layer2
            for j in range(len(outputU)):
                deltao = deltao + outputU[j]*neth2[j][i]
            weight=weighth[i]
            weighth[i] = weighth[i] + (momentum * weighthL[i]) + (learn * deltao)
            weighthL[i]=weight
            deltao=0
        # print("wUp-----------------------------")
        # print(str(len(weighthU)))

    def deltaWHidden1(arg, node, weighthL,weighth,momentum,learn, deltah2, neth1):
        unith2 = 0;
        weight = 0;
        print(str(weighthL[0][0]) + "----")
        print(str(learn) + "++++++")
        print(str(unith2) + "======")
        for i in range(node):  # perulangan sebanyak node hidden layer1
            for j in range(node):  # perulangan sebanyak node hidden layer2
                for k in range(len(deltah2)):
                    unith2 = unith2 + deltah2[k][i]*neth1[k][i]
                weight = weighth[i][j]
                weighth[i][j] = weighth[i][j] + (momentum * weighthL[i][j]) + (learn * unith2)
                weighthL[i][j] = weight
                unith2=0


        print("wUp-----------------------------")
        # print(str(len(weighthU)))



    def deltaWIn(arg, node, weighthL, weighth, momentum, learn, deltah1, neth1):
        unith2 = 0;
        net = 0
        weight = 0;
        for i in range(node):  # perulangan sebanyak node hidden layer1
            for j in range(len(weighth)):  # perulangan sebanyak node hidden layer2
                for k in range(len(deltah1)):
                    unith2 = unith2 + deltah1[k][i]*neth1[k][i]
                weight = weighth[i][j]
                weighth[i][j] = weighth[i][j] + (momentum * weighthL[i][j]) + (learn * unith2)
                weighthL[i][j] = weight
                unith2 = 0

        # print("wUp-----------------------------")
        # print(str(len(weighthU)))

    def updateBias(arg, node, biasO, biasHidden1,biasHidden2, learn,momentum, deltaO, deltah1, deltah2):
        bias = 0
        net = 0
        delta = 0
        for i in range(len(deltaO)):
            delta = delta+ (learn * deltaO[i])
        bias = biasO + momentum* biasO + delta
        biasO = bias
        delta = 0

        for i in range(node):  # perulangan sebanyak node hidden layer1
            for j in range(len(deltah2)):
                delta = delta + (learn* deltah2[j][i])
            bias = biasHidden2[i] + momentum * biasHidden2[i] +  delta
            biasHidden2[i] = bias
            delta = 0
        delta = 0

        for i in range(node):  # perulangan sebanyak node hidden layer1
            for j in range(len(deltah1)):
                delta = delta + (learn* deltah1[j][i])
            bias = biasHidden1[i] + momentum * biasHidden1[i] + delta
            biasHidden1[i] = bias
            delta = 0


        # print("BIAS-----------------------------")
        # print(str(len(biasHidden1)))
        # print(str(len(biasHidden2)))
        # print(str(biasO))

    def pengujian(self,update):
        t.refresh()
        t.hitungH1(t.wInput, t.nodeHidden, t.test, t.hw1, t.h1, t.nodeinput, t.biasHidden1,update)  # layer 1
        t.hitungH2(t.wHidden1, t.nodeHidden, t.h1, t.hw2, t.h2, t.biasHidden2, update)  # layer2
        t.hitungOutput(t.wHidden2, t.nodeHidden, t.h2, t.o, t.outputNet, t.biasO, update)
        a=0
        kelasB=0
        for i in range(len(t.outputNet)):
            if(t.outputNet[i] <= 0.5):
                a = 0
                print("Data ke-"+str(i+1)+" "+str(t.outputNet[i]) + " Kelas : "+ str(a) +" Kelas sebelum : " + str(t.testLabel[i]))
            elif (t.outputNet[i] > 0.5):
                a = 1
                print("Data ke-"+str(i+1)+" "+str(t.outputNet[i]) + " Kelas : " + str(a) + " Kelas sebelum : " + str(t.testLabel[i]))
            if(a == t.testLabel[i]):
                kelasB = kelasB + 1
            print(str(a))
        Akurasi = kelasB/(len(t.test)) * 100
        print ("Akurasi:" +str(Akurasi) +"%")


    #def feedForward(arg, train, nodeHidden, wInput, hw1, h1,wHidden1,wHidden2, hw2, h2,o,outputNet,index,jumFitur):
    def feedForward(self,update):
        t.refresh()
        t.hitungH1(t.wInput, t.nodeHidden, t.train , t.hw1, t.h1, t.nodeinput, t.biasHidden1,update)#layer 1
        t.hitungH2(t.wHidden1, t.nodeHidden, t.h1, t.hw2, t.h2,t.biasHidden2,update)#layer2
        t.hitungOutput(t.wHidden2, t.nodeHidden, t.h2, t.o,t.outputNet,t.biasO,update)
        t.errorO = t.error(t.trainLabel, t.outputNet,t.errorO)
        print("Loss: "+str(t.errorO))

    def backwardSweep(arg,update):  # menghitung nilai h
        t.deltaOutput(t.trainLabel, t.outputNet, t.outputUnit,update)
        # print(str(t.outputNet))
        t.deltaHidden2(t.h2, t.nodeHidden, t.outputNet, t.wHidden2, t.hUnit2,update)
        t.deltaHidden1(t.h1, t.nodeHidden, t.hUnit2, t.wHidden1, t.hUnit1,update)
        print("----"+str(t.wHidden1[0][0]))
        print("----" + str(t.wHidden2[0]))
        t.deltaWHidden2(t.nodeHidden, t.wHid2L, t.wHidden2,t.momentum, t.learning, t.outputUnit,t.h2)
        print("+----" + str(t.wHidden2[0]))
        print("=----" + str(t.wHid2L[0]))
        t.deltaWHidden1(t.nodeHidden, t.wHid1L, t.wHidden1, t.momentum, t.learning, t.hUnit2, t.h1)
        print("+----" + str(t.wHidden1[0][0]))
        print("=----" + str(t.wHid1L[0][0]))
        t.deltaWIn(t.nodeHidden, t.wInL, t.wInput, t.momentum, t.learning, t.hUnit1, t.train )
        t.updateBias(t.nodeHidden, t.biasO, t.biasHidden1, t.biasHidden2, t.learning, t.momentum, t.outputUnit, t.hUnit1,t.hUnit2)
        t.pengujian(update)


    def refresh(self):
        t.hw1 = []
        t.hw2 = []
        t.h1 = []
        t.h2 = []
        t.o = []
        t.outputNet = []
        t.outputUnit = []
        t.hUnit1 = []
        t.hUnit2 = []


t = annImage()

t.jumTrain = t.loadData(t.trainLabel, t.jumTrain, "ann_dataset/facedatatrainlabels")
# print (t.trainLabel)
# print (t.jumTrain)
t.jumTest=t.loadData(t.testLabel, t.jumTest, "ann_dataset/facedatatestlabels")
# print (t.testLabel)
# print (t.jumTest)
t.load(t.train, "ann_dataset/facedatatrain")
# print (t.train)
t.load(t.test, "ann_dataset/facedatatest")
# print (t.test)
t.biasO = t.bobotAwal(t.nodeinput,t.nodeHidden,t.wInput,t.wHidden1,t.wHidden2,t.wInL,t.wHid1L,t.wHid2L,t.biasHidden1,t.biasHidden2, t.biasO)
#t.feedForward(t.train, t.nodeHidden, t.wInput, t.hw1, t.h1,t.wHidden1,t.wHidden2, t.hw2, t.h2,t.o,t.outputNet,0,t.jumFitur)
stop = False
update = False
n=0
# while (stop==False):
for i in range(500):
    print("Epoch :"+ str(i+1))
    t.feedForward(update)
    t.backwardSweep(update)
    n= n+1

    # update= True
    # if(t.errorO <= t.errorMax or n > t.epoch):
    #     stop= True






