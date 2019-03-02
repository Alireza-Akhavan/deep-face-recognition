import tensorflow as tf
import numpy as np
from detection.mtcnn import detect_face
from scipy import misc

default_color = (0, 255, 0) #BGR
default_thickness = 2
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

margin = 44
image_size = 160

class Detection:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.session, None)

    def detect(self, img, detect_multiple_faces = True):
        bboxes = []
        bounding_boxes, points = detect_face.detect_face(
                img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                bboxes.append(bb)
        return bboxes
            


    def align(self, img, detect_multiple_faces = True):
        faces = []
        bboxes = self.detect(img,False)         
        for bb in bboxes:
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            faces.append(scaled)
        return faces
    
    def crop_detected_face(self, img, bb):
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        return scaled
