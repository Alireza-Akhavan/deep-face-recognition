## onvert112_112to112_96.py
"convert112_112to112_96.py" is used to change aligned images size from 112x112 to 112x96.

### simple usage

1- Download Asian-Celeb dataset from [1](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) or [2](https://www.dropbox.com/s/5cd1ppfqprjluaq/faces_glintasia.zip?dl=0)

1- clone InsightFace_TF-master

2- python rec2image.py —include /mnt/data/asia/faces_glintasia —output /mnt/data/glintasia

3- python convert112_112to112_96.py /mnt/data/glintasia/ /mnt/data/glintasia_112-96
