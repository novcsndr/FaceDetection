import glob
import os



class ann:
    height = 70
    width=60
    nodeinput =height*width
    jumTrain=0
    jumTest=0
    trainLabel =[]
    train = []
    testLabel = []
    test = []
    wHidden1 =[]
    wHidden2 = []
    wHid1Update = []
    wHid2Update = []


    def loadData(arg,table,length,file):#memasukkan data label pada list
        f = open(file, "r")
        for line in f:
            table.append(line.strip().split("\n"))
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
        print (table)

    def bobotAwal(self):  # inisialisasi bobot Awal
        for i in Training.nodeinput:

t = Training()
t.jumTrain=t.loadData(t.trainLabel, t.jumTrain, "ann_dataset/facedatatrainlabels")
print (t.trainLabel)
print (t.jumTrain)
t.jumTest=t.loadData(t.testLabel, t.jumTest, "ann_dataset/facedatatestlabels")
print (t.testLabel)
print (t.jumTest)
t.load(t.train, "ann_dataset/facedatatrain")
print (t.train)
t.load(t.test, "ann_dataset/facedatatest")
print (t.test)







