# Deep Face Recognition using Tensorflow workshop material 

### Slides:
[slides](https://docs.google.com/presentation/d/10xwd5wik8xrSB_Eh2Pb213tf9RhITt5IyTrwP8htYm0/edit?usp=sharing)

### Download videos:
[Videos](http://class.vision/deep-face-recognition/)

## Important link

### facenet pretrained model (Tensorflow)
[davidsandberg/facenet/](https://github.com/davidsandberg/facenet/)

### Sphereface or Angular Softmax (Caffe)
[wy1iu/sphereface](https://github.com/wy1iu/sphereface)

### ArcFace (MXNet)
[deepinsight/insightface](https://github.com/deepinsight/insightface)

### AMSoftmax (Caffe)
[happynear/AMSoftmax](https://github.com/happynear/AMSoftmax)

### Iranian Face Dataset
[iran-celeb.ir](http://iran-celeb.ir)


____________________
 # Datasets

1. [CASIA WebFace Database](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). 10,575 subjects and 494,414 images
2. [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).13,000 images and 5749 subjects
3. [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/) 202,599 images and 10,177 subjects. 5 landmark locations, 40 binary attributes.
4. [MSRA-CFW](http://research.microsoft.com/en-us/projects/msra-cfw/). 202,792 images and 1,583 subjects.
5. [MegaFace Dataset](http://megaface.cs.washington.edu/) 1 Million Faces for Recognition at Scale
690,572 unique people
6. [FaceScrub](http://vintage.winklerbros.net/facescrub.html). A Dataset With Over 100,000 Face Images of 530 People.
7. [FDDB](http://vis-www.cs.umass.edu/fddb/).Face Detection and Data Set Benchmark. 5k images.
8. [AFLW](https://lrs.icg.tugraz.at/research/aflw/).Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. 25k images.
9. [AFW](http://www.ics.uci.edu/~xzhu/face/). Annotated Faces in the Wild. ~1k images.
10.[3D Mask Attack Dataset](https://www.idiap.ch/dataset/3dmad). 76500 frames of 17 persons using Kinect RGBD with eye positions (Sebastien Marcel)
11. [Audio-visual database for face and speaker recognition](https://www.idiap.ch/dataset/mobio).Mobile Biometry MOBIO http://www.mobioproject.org/
12. [BANCA face and voice database](http://www.ee.surrey.ac.uk/CVSSP/banca/). Univ of Surrey
13. [Binghampton Univ 3D static and dynamic facial expression database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html). (Lijun Yin, Peter Gerhardstein and teammates)
14. [The BioID Face Database](https://www.bioid.com/About/BioID-Face-Database). BioID group
15. [Biwi 3D Audiovisual Corpus of Affective Communication](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html).  1000 high quality, dynamic 3D scans of faces, recorded while pronouncing a set of English sentences.
16. [Cohn-Kanade AU-Coded Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm).  500+ expression sequences of 100+ subjects, coded by activated Action Units (Affect Analysis Group, Univ. of Pittsburgh.
17. [CMU/MIT Frontal Faces ](http://cbcl.mit.edu/software-datasets/FaceData2.html). Training set:  2,429 faces, 4,548 non-faces; Test set: 472 faces, 23,573 non-faces.
18. [AT&T Database of Faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) 400 faces of 40 people (10 images per people)

 
# Other Face Dataset

## Face Detection Dataset

### FDDB

paper: http://vis-www.cs.umass.edu/fddb/fddb.pdf

dataset: http://vis-www.cs.umass.edu/fddb/index.html#download


### Wider Face

extreme scale

paper: https://arxiv.org/pdf/1511.06523.pdf

dataset: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html

### MAFA

occlusion

paper: http://openaccess.thecvf.com/content_cvpr_2017/papers/Ge_Detecting_Masked_Faces_CVPR_2017_paper.pdf

dataset: http://www.escience.cn/people/geshiming/mafa.html

### 4k face dataset

hight resolution

paper: https://arxiv.org/pdf/1804.06559.pdf

### Unconstrained Face Detection Dataset (UFDD)

different weather

paper: https://arxiv.org/abs/1804.10275
	
dataset: https://github.com/hezhangsprinter/UFDD

### wildest faces

paper: https://arxiv.org/pdf/1805.07566.pdf

### Multi-Attribute Labelled Faces (MALF)

paper: http://www.cbsr.ia.ac.cn/faceevaluation/faceevaluation15.pdf

dataset: http://www.cbsr.ia.ac.cn/faceevaluation/#reference

### IJB-A Dataset
paper: https://zhaoj9014.github.io/pub/IJBA_1N_report.pdf

dataset: https://www.nist.gov/itl/iad/image-group/ijb-dataset-request-form

# Age Estimation Dataset

### Adience dataset
dataset: https://talhassner.github.io/home/projects/Adience/Adience-data.html

statistic:
Total number of images: 26,580
Total number of subjects: 2,284
Number of age groups: 8 (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-)
Gender labels: Yes
In the wild: Yes
Subject labels: Yes 

### UTK-Face
dataset: https://susanqq.github.io/UTKFace/

### APPA-REAL (real and apparent age)
paper: http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w48/Clapes_From_Apparent_to_CVPR_2018_paper.pdf

dataset: http://chalearnlap.cvc.uab.es/dataset/26/description/


# Face Landmark Detection Dataset

### 300W
paper: https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf

### COFW

occluded to different degrees

paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/12/BurgosArtizzuICCV13rcpr.pdf
### AFLW

faces with large head pose up to 120◦ for yaw and 90◦ for pitch and roll.

paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf

dataset: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/

### WFLW

from wider face dataset

paper: https://arxiv.org/pdf/1805.10483.pdf

dataset: https://wywu.github.io/projects/LAB/WFLW.html


# papers:
[Deep Face Recognition (2015)](http://cis.csuohio.edu/~sschung/CIS660/DeepFaceRecognition_parkhi15.pdf)

[FaceNet: A Unified Embedding for Face Recognition and Clustering (2015)](https://arxiv.org/abs/1503.03832)

[A Discriminative Feature Learning Approach for Deep Face Recognition (2016)](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31)

[SphereFace: Deep Hypersphere Embedding for Face Recognition (2018)](https://arxiv.org/abs/1704.08063)

[Additive Margin Softmax for Face Verification (2018)](https://arxiv.org/abs/1801.05599)

[Ring loss: Convex Feature Normalization for Face Recognition(2018)](https://arxiv.org/abs/1803.00130)

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition (2019)](https://arxiv.org/abs/1801.07698)

[Deep Face Recognition: A Survey (2019)](https://arxiv.org/pdf/1804.06655)

#### source: 
	https://github.com/jian667/face-dataset

	https://github.com/jian667/Face-Resources/
