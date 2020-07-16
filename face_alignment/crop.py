import cv2
import numpy as np
import os
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector



def process(filename, type, output_size, output_folder, dst_name=None):
    # print("processing: ", filename)
    img = cv2.imread(filename)
    try:
        bbox, facial5points = detector.detect_faces(img)
    except ValueError:
        print("error**********************************")
        return

    
    # # # show ref_5pts
    tmp_pts = np.array([[38.29459953, 73.53179932, 56.02519989, 41.54930115, 70.72990036, 51.69630051, 51.50139999,  71.73660278,  92.3655014,
         92.20410156]])
    empty_face = np.zeros((112, 112, 3))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    if (len(bbox) > 0):
        if type == 'labeled':
            # find the max bbox
            max_bb = []
            for box in bbox:
                x, y, r, b, _ = list(map(int, box))
                w = r - x + 1
                h = b - y + 1
                max_bb.append(w * h)
            index = max_bb.index(max(max_bb))
            facial5points = facial5points[[index]]
            facial5point = np.reshape(facial5points, (2, 5))
            dst_img = warp_and_crop_face(
                img, facial5point, reference_pts=reference_5pts, crop_size=output_size)

            cv2.imwrite('{}/{}_mtcnn_aligned_{}x{}.jpg'.format(
                output_folder, dst_name, output_size[0], output_size[1]), dst_img)
        elif type == 'unlabelled':
            for i in range(bbox.shape[0]):
                facial5point = np.reshape(facial5points[i], (2, 5))
                dst_img = warp_and_crop_face(img, facial5point, reference_pts=reference_5pts, crop_size=output_size)

                cv2.imwrite('{}/{}_{}_mtcnn_aligned_{}x{}_{}.jpg'.format(
                    output_folder, dst_name, type, output_size[0], output_size[1], i), dst_img)


if __name__ == "__main__":
    detector = MtcnnDetector()
    # folder = "/home/coin/Documents/face_recognition/chapter3/lfw_deepblue"
    # output_folder = '/home/coin/Documents/face_recognition/chapter3/data/labeled'
    # type = 'labeled'

    folder = "/home/coin/Documents/face_recognition/chapter3/CASIA-maxpy-clean"
    output_folder = '/home/coin/Documents/face_recognition/chapter3/data/unlabelled'
    type = 'unlabelled'



    for name_folder in os.listdir(folder):
        sub_folder = folder + '/' + name_folder
        if name_folder[0] == 'A' or name_folder == "0000045":
            for filename in os.listdir(sub_folder):
                src_file = sub_folder + '/' + filename

                basename = os.path.basename(src_file)
                basename = basename[: basename.rfind('.')]
                if type == "labeled":
                    dst_name = basename
                else:
                    dst_name = name_folder + '_' + basename
    
                process(src_file, type, output_size=(112, 112), output_folder=output_folder, dst_name=dst_name)
