import numpy as np
import pickle
import os

r = np.random.RandomState(0)
#Tipe penyimpanan data array numpy
#Untuk program ini terdapat beberapa fungsi untuk menghitung fungsi aktivasi, loss function, delta pada output atau
# hidden layer, juga update bias dan weight. Pada program ini juga akan melakukan testing berdasarkan sebanyak epoch
# yang sudah dinisiasi. dan kemudian bila ada model yang duah tersimpan akan langsung dilakukan pengujian

class ann_sigmoid:
    #nilai atribut yang digunakan
    jumlahTraining = 752
    jumlahTesting = 150
    nodeHidden = 100
    momentum = 0.3
    learning = 0.5/752
    nodeinput = 4200
    nodeoutput = 1
    errorMax = 0.001
    epoch = 500
    maxerror = 0.001


    def loadData(arg, file):  # memasukkan data label test/training pada array numpy
        table = []
        f = open(file, "r")
        b = 0
        for line in f:
            a = line.strip().split("\n")
            if a[0] == '1':
                b = 1
            elif a[0] == '0':
                b = 0
            table.append(b)
        return np.array(table)

    def load(arg, file):  # mengubah dataset test/training ke dalam biner pada array numpy
        f = open(file, "r")
        biner = []
        table = []
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
        return np.array(table)

    def bobotAwal(arg, node1, node2, nodeinput):  # inisialisasi bobot Awal
        bound = 4 * np.sqrt(6.0 /(node1+nodeinput))
        # print(bound)
        weight = []
        for i in range(node1):  # perulangan sebanyak node hidden layer
            weight.append(r.uniform(-bound, bound, node2))
        return np.array(weight)

    def bobotBias(arg, node1, nodeinput):  # inisialisasi bobot Awal bias
        bound = 4 * np.sqrt(6.0/(node1+nodeinput))
        weight = []
        # for i in range(node1):  # perulangan sebanyak node hidden layer
        weight.append(r.uniform(-bound, bound, node1))
        print(weight)
        return np.array(weight)

    def layerAktivasi(arg, nilaiNode, weight, bias):  # menghitung nilai fungsi aktivasi tiap layer
        net = np.dot(nilaiNode, weight)+bias
        sigmoid = 1 / (1+ (np.exp(-net)))
        return np.array(sigmoid)

    def error(arg, target, aktivasioutput):  # menghitung nilai error atau loss
        error = np.sum(0.5 * (np.power((aktivasioutput-target), 2)))
        return error

    def deltaOutput(arg, target, aktivasioutput):  # menghitung nilai delta output layer
        deltaO = (aktivasioutput-target) * (aktivasioutput*(1-aktivasioutput))
        return np.array(deltaO)

    def deltaHidden(arg, aktivasiHidden, deltaOutput, weight):  # menghitung nilai delta hidden layer
        sigma = np.dot(weight, np.transpose(deltaOutput))
        temphdelta = ((1 - aktivasiHidden)*aktivasiHidden)
        temphdelta = np.transpose(sigma) * temphdelta
        return np.array(temphdelta)

    def weightUpdate(arg, weightLama, weight, momentum, learn, delta, aktivasi):  # menghitung perubahan weight#aktivasi=input
        weightU = weight +((-learn * np.dot(np.transpose(aktivasi), delta)) + (momentum * weightLama))
        weightLama =(-learn * np.dot(np.transpose(aktivasi), delta))+ (momentum * weightLama)
        return np.array(weightU), np.array(weightLama)

    def biasUpdate(arg, biasLama, bias, momentum, learn, delta):  # menghitung perubahan bias
        #tanpa momentum
        #biasU = bias + momentum * biasLama + learn * delta.sum(axis=0)
        #dengan momtntum
        biasU = bias + ((-learn * delta.sum(axis=0)) + (momentum * biasLama))
        biasLama= ((-learn * delta.sum(axis=0)) + (momentum * biasLama))
        return np.array(biasU), np.array(biasLama)

    # def feedfoorward(arg, train, wInput, biasHidden1, wHidden1, biasHidden2, wHidden2, biasO, trainLabel):
    #     aktivasiH1 = t.layerAktivasi1(train, wInput, biasHidden1)
    #     aktivasiH2 = t.layerAktivasi2(aktivasiH1, wHidden1, biasHidden2)
    #     aktivasiO = t.layerAktivasi3(aktivasiH2, wHidden2, biasO)
    #     loss = t.error(trainLabel, aktivasiO)
    #     print ("loss:" + str(loss))
    #     return aktivasiH1, aktivasiH2, aktivasiO, loss
    #
    # def backpropagation(arg, trainLabel, aktivasiO, aktivasiH2, aktivasiH1, wHidden1, wHidden2, wInput, wHidden2Lama,
    #     wHidden1Lama, wInputLama, momentum, learning, train, biasOLama, biasO):
    #     deltaO = t.deltaOutput(trainLabel, aktivasiO)
    #     deltaHiddenLayer2 = t.deltaHidden(aktivasiH2, deltaO, wHidden2)
    #     deltaHiddenLayer1 = t.deltaHidden(aktivasiH1, deltaHiddenLayer2, wHidden1)
    #
    #     wH2, wHidden2L = t.weightUpdate(wHidden2Lama, wHidden2, momentum, learning, deltaO, aktivasiH2)
    #     wH1, wHidden1L = t.weightUpdate(wHidden1Lama, wHidden1, momentum, learning, deltaHiddenLayer2, aktivasiH1)
    #     wI, wInputL = t.weightUpdate(wInputLama, wInput, momentum, learning, deltaHiddenLayer1, train)
    #     bO, biasOL = t.biasUpdate(biasOLama, biasO, momentum, learning, deltaO)
    #     biasH2, biasH2L = t.biasUpdate(wHidden1Lama, wHidden1, momentum, learning, deltaHiddenLayer2)
    #     biasH1, biasH1L = t.biasUpdate(wInputLama, wInput, momentum, learning, deltaHiddenLayer1)
    #     return wH2, wH1, wI, bO, biasH2, biasH1, biasOL, biasH2L, biasH1L, wInputL, wHidden1L, wHidden2L

    def pengujian(arg, test, wInput, biasHidden1, wHidden1, biasHidden2, wHidden2, biasO, testLabel):#pengujian
        # aktivasiH1, aktivasiH2, aktivasiO, loss = t.feedfoorward(test, wInput, biasHidden1,wHidden1, biasHidden2,wHidden2, biasO, testLabel)
        aktivasiH1P = t.layerAktivasi(test, wInput, biasHidden1)#menghitung fungsi aktivasi tiap layer untuk mendapat outputnya
        aktivasiH2P = t.layerAktivasi(aktivasiH1P, wHidden1, biasHidden2)
        aktivasiOP = t.layerAktivasi(aktivasiH2P, wHidden2, biasO)
        loss = t.error(testLabel, aktivasiOP)
        # print ("loss:" + str(loss))
        count = 0
        kelas = 0
        akurasi = 0.0
        # print(aktivasiOP)
        # print(aktivasiOP.shape)
        for i in range(t.jumlahTesting):
            if (aktivasiOP[i][0] > 0.5):
                kelas = 1
            elif (aktivasiOP[i][0] <= 0.5):
                kelas = 0
            if (kelas == testLabel[i]):
                count = count + 1
            # print("Data ke-" + str(i + 1) + " " + str(aktivasiOP[i][0]) + " Kelas : " + str(kelas) + " Kelas sebelum : " + str(testLabel[i]))

        akurasi = float (count)/t.jumlahTesting* 100

        return float(akurasi)

    def saveModel(arg, data, filename):
        with open(filename, 'wb') as save:
            pickle.dump(data, save)

    def loadModel(arg, filename):
        with open(filename, 'rb') as data:
            result = pickle.load(data)
        return result

    def gantiBobot(arg, wH2, wH1, wI, wHidden2L, wHidden1L, wInputL, biasOL, biasH2L,biasH1L, biasO, biasH2, biasH1):
        return  wH2, wH1, wI, wHidden2L, wHidden1L, wInputL, biasOL, biasH2L,biasH1L, biasO, biasH2, biasH1


t = ann_sigmoid()
#disi pemilihan data set 1
# load data
trainLab = t.loadData("ann_dataset/facedatatrainlabels")
trainLabel = np.zeros((752, 1))
for i in xrange(trainLabel.shape[0]):
    trainLabel[i] = trainLab[i]
testLabel = t.loadData("ann_dataset/facedatatestlabels")
train = t.load("ann_dataset/facedatatrain")
test = t.load("ann_dataset/facedatatest")
# random bobot dan bias
wInput = t.bobotAwal(t.nodeinput, t.nodeHidden, t.nodeinput)

wHidden1 = t.bobotAwal(t.nodeHidden, t.nodeHidden, t.nodeinput)
wHidden2 = t.bobotAwal(t.nodeHidden, t.nodeoutput, t.nodeinput)
biasHidden1 = t.bobotBias(t.nodeHidden, t.nodeinput)
biasHidden2 = t.bobotBias(t.nodeHidden, t.nodeinput)
biasO = t.bobotBias(t.nodeoutput, t.nodeinput)
# set array untuk menampung bobot dan bias sebelum diupdate
# print (test.shape)
wInputLama = np.zeros((4200, 100))
wHidden1Lama = np.zeros((100, 100))
wHidden2Lama = np.zeros((100, 1))
biasHidden1Lama = np.zeros((1, 100))
biasHidden2Lama = np.zeros((1, 100))
biasOLama = np.zeros((1, 1))
n = 0
stop = False
# while (stop==False):
for i in range(t.epoch):
    # biasHidden1,biasHidden2,biasO = t.repeatBroadcast(biasHidden1,biasHidden2,biasO,t.jumlahTraining)
    print("Epoch: " + str(i + 1))
    # if (i > 0):  # melakukan perubahan nilai variable bobot dan bias yang telah diupdate
    #     wHidden2, wHidden1, wInput, wHidden2Lama,  wHidden1Lama, wInputLama, biasOLama, biasHidden2Lama, biasHidden1Lama, biasO,  biasHidden2,  biasHidden1 = t.gantiBobot(
    #         wH2, wH1, wI, wHidden2L, wHidden1L, wInputL, biasOL, biasH2L,biasH1L, biasO, biasH2, biasH1)
    # aktivasiH1, aktivasiH2, aktivasiO, loss = t.feedfoorward(train, wInput, biasHidden1, wHidden1, biasHidden2, wHidden2, biasO,trainLabel)
    # wHidden2, wHidden1, wInput, biasO, biasHidden2, biasHidden1, biasOLama, biasHidden2Lama, biasHidden1Lama, wInputLama, wHidden1Lama, wHidden2Lama= t.backpropagation(trainLabel, aktivasiO, aktivasiH2, aktivasiH1, wHidden1, wHidden2, wInput, wHidden2Lama,
    #                    wHidden1Lama, wInputLama, t.momentum, t.learning, train, biasOLama, biasO)
    # feed forward
    aktivasiH1 = t.layerAktivasi(train, wInput, biasHidden1)  # memanggil fungsi aktivasi input-hidden layer1
    aktivasiH2 = t.layerAktivasi(aktivasiH1, wHidden1, biasHidden2)  # memanggil fungsi aktivasi hidden layer1-hidden layer2
    aktivasiO = t.layerAktivasi(aktivasiH2, wHidden2,  biasO)  # memanggil fungsi aktivasi hidden layer2-outputlayer
    loss = t.error(trainLabel, aktivasiO)  # menghitung error
    print ("loss:" + str(loss))
    # back propagation
    deltaO = t.deltaOutput(trainLabel, aktivasiO)  # memanggil fungsi menghitung delta output
    deltaHiddenLayer2 = t.deltaHidden(aktivasiH2, deltaO, wHidden2)  # memanggil fungsi menghitung delta hidden layer2
    deltaHiddenLayer1 = t.deltaHidden(aktivasiH1, deltaHiddenLayer2, wHidden1)  # memanggil fungsi menghitung delta hidden layer1
    # melakukan update nilai weight dan bias
    wHidden2, wHidden2Lama = t.weightUpdate(wHidden2Lama, wHidden2, t.momentum, t.learning, deltaO, aktivasiH2)
    wHidden1, wHidden1Lama = t.weightUpdate(wHidden1Lama, wHidden1, t.momentum, t.learning, deltaHiddenLayer2,aktivasiH1)
    wInput, wInputLama = t.weightUpdate(wInputLama, wInput, t.momentum, t.learning, deltaHiddenLayer1, train)
    biasO, biasOLama = t.biasUpdate(biasOLama, biasO, t.momentum, t.learning, deltaO)
    biasHidden2, biasHidden2Lama = t.biasUpdate(biasHidden2Lama, biasHidden2, t.momentum, t.learning, deltaHiddenLayer2)
    biasHidden1, biasHidden1Lama = t.biasUpdate(biasHidden1Lama, biasHidden1, t.momentum, t.learning, deltaHiddenLayer1)
    akurasi = t.pengujian(test, wInput, biasHidden1, wHidden1, biasHidden2, wHidden2, biasO, testLabel)
    print("Akurasi : " + str(akurasi) + "%")

dir_path = os.path.exists('model/wHidden2')
if(dir_path is True):
    print("----------------Testing----------------")
    t.loadModel("model/wHidden2")
    t.loadModel("model/wHidden1")
    t.loadModel("model/wInput")
    t.loadModel("model/biasO")
    t.loadModel("model/biasHidden2")
    t.loadModel("model/biasHidden1")
    akurasi = t.pengujian(test, wInput, biasHidden1, wHidden1, biasHidden2, wHidden2, biasO, testLabel)
    # print("Akurasi : " + str(akurasi) + "%")

# elif(dir_path is False):
#     t.saveModel(wH2, "model/wHidden2")
#     t.saveModel(wH1, "model/wHidden1")
#     t.saveModel(wI, "model/wInput")
#     t.saveModel(bO, "model/biasO")
#     t.saveModel(biasH2, "model/biasHidden2")
#     t.saveModel(biasH1, "model/biasHidden1")