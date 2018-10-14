# Classifying TCGA cancer types from gene expression data using the
# fastai deep learning library

This is a rough exploratory project which is a follow-up to an earlier attempt
to use pytorch to build a basic 2D convolutional neural network to attempt to
accurately classify the cancer type of patient samples from The Cancer Genome
Atlas (TCGA) using RNAseq gene expression data for each sample as dependent
variables for predicting the dependent variable of cancer type. That notebook
has been converted into the tcga_fpkm13class_convnet.py python file found in
this repo.

TCGA RNAseq gene expression data in the form of one FPKM
(fragments per kilobase per million mapped reads) value per gene was downloaded
for 13 different cancer types from the TCGA data portal, and post-processed
to match the TCGA patient barcode with the RNAseq file UUID, allowing each gene
expression file to be matched with the known cancer type for that patient.
The 13 cancer types chosen all had at least 100 patient samples for that cancer
type, with the exception of GBM (glioblastoma multiforme) which has 95 patients.
This dataset has a total of 4287 samples, which were divided into the training,
validation, and test sets.

The overall approach was to fine-tune a 2D convolutional neural
network (resnet34 and densenet121) pre-trained on ImageNet image data to learn
to correctly classify gene expression signatures represented as 2D matrices of
values, formatted similarly to a greyscale image to take advantage of the
existing pre-trained models available in the fastai library.
Each text file containing gene expression values as float for a single patient
sample was read into a numpy array in a previous pytorch project, then saved as
a jpeg image file, which enabled use of the fastai ImageClassifierData
dataloader.

### Results

A final overall accuracy of ~91% was achieved across all 13 classes, with varying
degrees of accuracy per class. One cancer type, READ (Rectal adenocarcinoma) had
low accuracy due to consistently being mis-classified as COAD
(Colon adenocarcinoma), however the anatomical proximity of these cancers, both
of which are adenocarcinomas, suggests that these two cancer types may be quite
similar in terms of gene expression signature.

### Class Activation Mapping (CAM)

The second part of this notebook explores which genes in each sample
are most responsible for classification by attempting a technique called
class activation mapping (CAM), where the highest activations for each "pixel"
(gene) in the last convolutional layer of the assigned class filter are mapped
to over the original "image", indicating which parts of the image
(and therefore which genes) are used to predict the class of the sample.

Jonathan Irish 2018-10-14
