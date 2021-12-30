import random
import math
# import numpy as np

class ann:
    height = 70
    width=60
    nodeHidden = 100
    momentum=0.3
    learning=0.5
    jumFitur = 4200
    nodeinput =height*width
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
    wInUpdate = []
    wHid1Update = []
    wHid2Update = []
    biasHidden1 = []
    biasHidden2 = []
    biasO = 0

    def loadData(arg, table, length, file):  # memasukkan data label pada list
        f = open(file, "r")
        b = 0
        for line in f:
            a = line.strip().split("\n")
            if a[0] == '1':
                b = 1
            elif a[0] == '0':
                b = 0
            table.append(b)
            length += 1
        return length

    def load(arg, table, file):  # mengubah dataset ke dalam biner dan disimpan dalam list
        f = open(file, "r")
        biner = []
        b = 0
        for line in f:
            tuple = line
            b += 1
            for i in tuple:
                if i == "#":
                    biner.append(1)
                elif i == "\n":
                    continue
                else:
                    biner.append(0)
            if b % 70 == 0:
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
        print(str(len(weighth)))

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
        print(str(len(weighth1)))

        random.seed(0)
        for m in range (node):  # perulangan sebanyak node hidden layer2
            a = random.uniform(-bound, bound)
            weighth2.append(a)
            weighth2U.append(0)
        biasO=random.uniform(-bound, bound)

        print(str(len(weighth2)))
        return biasO
    def hitungH1(arg,weighth, node, input, hnode,honode, fitur, jumTrain,biasHidden1,update):  # menghitung nilai h
        temphnode=0
        ho=0
        sigma=[]
        hidden =[]
        for k in range(jumTrain):  # perulangan sebanyak jumlsh data input
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

        print("h1------------------------------------")
        print(str(len(hnode)))
        print(str(len(honode)))

        print(weighth[0][1])
        print(weighth[1][1])


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
        print("h2------------------------------------")
        print(str(len(hnode)))
        print(str(len(honode)))


t = ann()

t.jumTrain = t.loadData(t.trainLabel, t.jumTrain, "ann_dataset/facedatatrainlabels")
# print (t.trainLabel)
print (t.jumTrain)
t.jumTest=t.loadData(t.testLabel, t.jumTest, "ann_dataset/facedatatestlabels")
# print (t.testLabel)
print (t.jumTest)
t.load(t.train, "ann_dataset/facedatatrain")
print (len(t.train))
t.load(t.test, "ann_dataset/facedatatest")
print (len(t.test))
t.load(t.train, "ann_dataset/facedatatrain")
# print (t.train)
t.load(t.test, "ann_dataset/facedatatest")
# print (t.test)
t.biasO = t.bobotAwal(t.nodeinput,t.nodeHidden,t.wInput,t.wHidden1,t.wHidden2,t.wInUpdate,t.wHid1Update,t.wHid2Update,t.biasHidden1,t.biasHidden2, t.biasO)
#t.feedForward(t.train, t.nodeHidden, t.wInput, t.hw1, t.h1,t.wHidden1,t.wHidden2, t.hw2, t.h2,t.o,t.outputNet,0,t.jumFitur)
stop = False
update = False
i=0