"""20180708JCI Experimenting with building 2D convolutional neural networks
   in pytorch using TCGA RNAseq gene expression data to attempt building
   a cancer type classifier which predicts cancer type from gene expression
   values for each sample represented as a 2D pytorch tensor similar to
   how a greyscale image would be loaded from torchvision: each "pixel"
   value is a gene expression value. The performance of this model will
   be compared to an earlier dense (fully-connected) neural net model.

   This python script was adapted from a Jupyter Notebook, and may contain
   code which makes sense only in the interactive Notebook environment."""

get_ipython().magic(u'pylab inline')
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Expand simpler binary classifier from lung2class to 13 classes representing
# 13 different cancer types

# numpy "image" array dimensions for gene expression data
# NOTE: 3 values are dropped from each patient sample to achieve integer dims
height = 225
width = 64800 / height

# Create a dataset class for the gene-fpkm values data (based on pytorch data processing tutorial:
# pytorch.org/tutorials/beginner/data_loading_tutorial.html)

# This version normalizes fpkm values from 0 to 255 for single channel greyscale "image" representation
class GeneExpressionFPKMDatasetN1(Dataset):
    """Gene expression RNAseq FPKM values dataset."""

    def __init__(self, samplenames_file, data_dir, transform=None):
        self.filename_frame = pd.read_csv(( os.path.join(data_dir
                                                        , samplenames_file
                                                        )
                                          )
                                         , sep="\s+"
                                         , names=["filename","label"]
                                         , dtype={ 'filename':'object'
                                                 , 'label':'category' }
                                         )
        self.root_dir = data_dir
        # 180518 store original class names from manifest file
        self.class_encoder = LabelEncoder().fit(self.filename_frame.label)
        self.transform = transform

    def __len__(self):
        return len(self.filename_frame)

    def __getitem__(self, idx):
        sample_name = os.path.join(self.root_dir, self.filename_frame.iloc[idx, 0])
        label = self.class_encoder.transform([self.filename_frame.iloc[idx, 1]])
        pxdf = pd.read_csv(sample_name, sep='\s+', header=None, names=['fpkm'])
        pxdf['normval'] = (pxdf.fpkm - pxdf.fpkm.mean()) / pxdf.fpkm.std()
        fpkmvals = pxdf.normval.astype(np.float32).values
        # TODO: add 0-padding to end of array so we don't lose 3 expression values
        fpkmvals = fpkmvals[3:] # drop first 3 vals to keep shape

        # NOTE: no data augmentation transforms (values are not loaded with torchvision)
        # torch image dim order: C X H X W
        fpkmvals = fpkmvals.reshape((1, 224, 270))
        fpkmvals = torch.from_numpy(fpkmvals)
        label = torch.from_numpy(label)

        return {'fpkmvals': fpkmvals.type(torch.FloatTensor),
                'label': label.type(torch.LongTensor)}

# 2D convolutional neural network pytorch model class
# NOTE: this version takes the input dims as an arg
# and calculates the total features output from the last conv2d
# layer for sizing the first fully-connected layer

class MyConv2DNet(nn.Module):
    """Basic 2D Convolutional neural network with fully connected final layers,
       a toy version of a vgg-style 2D conv model"""

    def __init__(self, inshape=(1,224,270)):
        super(MyConv2DNet, self).__init__()
        # 1 input channel for FPKM values, 12 filters, 3x3 conv
        self.conv1 = nn.Conv2d(1, 12, 3, padding=(1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.convdrop1 = nn.Dropout2d(p=0.3)

        conv_out_feats = self._get_conv_outshape(inshape)

        self.fc1 = nn.Linear(conv_out_feats, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 13) # 13 classes for cancer types represented in train/valid data

    # no softmax layer: pytorch nn.CrossEntropyLoss function does not
    # require a softmax layer in the model forward function
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dims except batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x

    def _get_conv_outshape(self, inshape):
        batchsz = 1
        dummy_input = Variable(torch.rand(batchsz, *inshape))
        dummy_output = self._conv_forward(dummy_input)
        feature_cnt = dummy_output.data.view(batchsz, -1).size(1)
        return feature_cnt




# In[6]:


# shuffle manifest file to try to ensure both classes represented in any given training batch
testdata = GeneExpressionFPKMDatasetN1("test_manifest.txt", "/ssdata/data/tcga/fpkm13class/test/")
traindata = GeneExpressionFPKMDatasetN1("train_manifest.txt", "/ssdata/data/tcga/fpkm13class/train/")
validdata = GeneExpressionFPKMDatasetN1("valid_manifest.txt", "/ssdata/data/tcga/fpkm13class/valid/")

# Data Loaders (data loaders create generators over instances of Dataset classes)
testloader = DataLoader(testdata, batch_size=8, shuffle=True, num_workers=4)
trainloader = DataLoader(traindata, batch_size=8, shuffle=True, num_workers=4)
validloader = DataLoader(validdata, batch_size=8, shuffle=False, num_workers=4)


# use wrappers to init and train model over several sets of epochs

# wrapper functions for net initialization, training runs, and quick accuracy check
# TODO: create cross-validation procedure and work on visualization of loss during training

def newNetCuda(model):
    net = model()
    net.training = True
    net.cuda()
    net.zero_grad()
    return net

def trainRun(model, epochs=10, lr=1.0e-3):
    net.training = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-6)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['fpkmvals']
            labels = data['label']
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
            labels = labels.squeeze() # must be Tensor of dim BATCH_SIZE, MUST BE 1Dimensional!
            optimizer.zero_grad()
            outputs = net(inputs) # calls "forward" method
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running stats
            running_loss += loss.data[0]
            if i % 8 == 7:
                print("[%d, %5d] loss: %.4f" %
                      (epoch + 1, i + 1, running_loss / 8))
                running_loss = 0.0
    print("Finished")


def checkAccuracy(model, tloader, vloader):
    net = model
    net.training = False
    trainloader = tloader
    validloader = vloader
    class_correctT = list(0. for i in range(13))
    class_totalT = list(0. for i in range(13))
    classes = ("BRCA","COAD","GBM","HNSC","KIRC","KIRP","LGG","LUAD","LUSC",
               "PRAD","READ","THCA","UCEC")
    for data in trainloader:
        imagesT, labelsT = data['fpkmvals'], data['label']
        outputsT = net(Variable(imagesT).cuda())
        _, predicted = torch.max(outputsT.data, 1)
        c = (predicted.cpu() == labelsT.squeeze()).numpy()
        for i in range(len(labelsT)):
            label = int(labelsT[i][0])
            class_correctT[label] += c[i]
            class_totalT[label] += 1
    print("Training set accuracy: ")    
    for i in range(len(classes)):
        print("Accuracy of %5s: %2d%%" % (classes[i], 100 * class_correctT[i] / class_totalT[i]))

    class_correctV = list(0. for i in range(13))
    class_totalV = list(0. for i in range(13))
    for data in validloader:
        imagesV, labelsV = data['fpkmvals'], data['label']
        outputsV = net(Variable(imagesV).cuda())
        _, predicted = torch.max(outputsV.data, 1)
        c = (predicted.cpu() == labelsV.squeeze()).numpy()
        for i in range(len(labelsV)):
            label = int(labelsV[i][0])
            class_correctV[label] += c[i]
            class_totalV[label] += 1
    print("Validation set accuracy: ") 
    for i in range(len(classes)):
        print("Accuracy of %5s: %2d%%" % (classes[i], 100 * class_correctV[i] / class_totalV[i]))

net = newNetCuda(model=MyConv2DNet)
get_ipython().magic(u'time trainRun(net, epochs=2)')

checkAccuracy(net, trainloader, validloader)

get_ipython().magic(u'time trainRun(net, epochs=7)')

checkAccuracy(net, trainloader, validloader)

trainRun(net, epochs=10, lr=1.0e-3)

checkAccuracy(net, trainloader, validloader)

torch.save(net.state_dict(), "/seq/tcga/predict_ctype/fpkm13class/tcga13class_convnet1_statedict.pth")

ctypeREADimgs = [d['fpkmvals'] for d in iter(validdata) if d['label'].numpy()[0] == 10]
ctypeBRCAimgs = [d['fpkmvals'] for d in iter(validdata) if d['label'].numpy()[0] == 0]

readimgs = [i.numpy() for i in ctypeREADimgs]
brcaimgs = [i.numpy() for i in ctypeBRCAimgs]

padzeros = np.zeros((224,270))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def imshow2(img):
    #img = img / 2 + 0.5
    imgarr = np.array([img[0,:,:], img[0,:,:], img[0,:,:]]).transpose(1,2,0)
    plt.imshow(imgarr)

def imshow3(img):
    #img = img / 2 + 0.5
    imgarr = np.array([img[0,:,:], padzeros, padzeros]).transpose(1,2,0)
    plt.imshow(imgarr)

first6 = readimgs[:6]

f, a = plt.subplots(figsize=(18,18))

for (n, i) in enumerate(first6):
    plt.gca()
    plotnum = 160 + (n + 1)
    plt.subplot(plotnum)
    imgarr = np.array([i[0,:,:], i[0,:,:], i[0,:,:]]).transpose(1,2,0)
    plt.imshow(imgarr)

firstbrca6 = brcaimgs[:6]

f, a = plt.subplots(figsize=(18,18))

for (n, i) in enumerate(firstbrca6):
    plt.gca()
    plotnum = 160 + (n + 1)
    plt.subplot(plotnum)
    imgarr = np.array([i[0,:,:], i[0,:,:], i[0,:,:]]).transpose(1,2,0)
    plt.imshow(imgarr)

firstbrca6 = brcaimgs[:6]

f, a = plt.subplots(figsize=(18,18))

for (n, i) in enumerate(firstbrca6):
    plt.gca()
    plotnum = 160 + (n + 1)
    plt.subplot(plotnum)
    imgarr = np.array([i[0,:,:], padzeros, padzeros]).transpose(1,2,0)
    plt.imshow(imgarr, interpolation="nearest")

# save fpkm tensors as images to test using image classifier libraries to load and use data
t1 = next(iter(testdata))

type(t1['fpkmvals'])

imgpath = '/ssdata/data/tcga/fpkm13class/fpkm_as_jpg'

torchvision.utils.save_image(t1['fpkmvals'], imgpath + '/test1.jpg', padding=0)

# function to save all tensors in dataset as jpg images with 
testdata = GeneExpressionFPKMDatasetN1("test_manifest.txt", "/ssdata/data/tcga/fpkm13class/test/")

t = next(iter(testdata))

testdata.class_encoder.inverse_transform(8)

t['fpkmvals'].shape

self.filename_frame = pd.read_csv((os.path.join(data_dir, samplenames_file)),
                                           sep="\s+", names=["filename","label"],
                                          dtype={'filename':'object', 'label':'category'})
        self.root_dir = data_dir
        self.labels = self.filename_frame['label'].factorize(sort=True)[0]

def save_dataset_to_jpgs(dataset, img_path=None):
    """Save Dataset instance to jpeg images with class labels as filename prefixes"""
    if img_path is None:
        img_path = dataset.root_dir
    for (i, e) in enumerate(dataset):
        lbl = dataset.class_encoder.inverse_transform(e['label'][0])
        jpg_fname = f'{lbl}_{i}.jpg'
        torchvision.utils.save_image(e['fpkmvals'], os.path.join(img_path, jpg_fname), padding=0)

save_dataset_to_jpgs(testdata, "/ssdata/data/tcga/testjpegs")

# create jpeg files of FPKM values for training and validation sets all in a single folder
# NOTE: validation set will be sampled from the training set when jpeg files are loaded into conv_learner
# dataset

alldata = GeneExpressionFPKMDatasetN1("alldata_manifest.txt",
                                      "/ssdata/data/tcga/fpkm13class/all/")

save_dataset_to_jpgs(alldata, img_path="/ssdata/data/tcga/fpkm13class/all/jpeg")
