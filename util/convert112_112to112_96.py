import imageio
import sys
import os
import argparse
import facenet
import random
from time import sleep
from tqdm import tqdm

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src_path,_ = os.path.split(os.path.realpath(__file__))
    dataset = facenet.get_dataset(args.input_dir)
    

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if args.random_order:
        random.shuffle(dataset)
    for cls in tqdm(dataset):
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if args.random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            #print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = imageio.imread(image_path)
                    nrof_successfully_aligned += 1
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)

                img_112_96 = img[:, 8:-8]
                imageio.imsave(output_filename, img_112_96)

                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
