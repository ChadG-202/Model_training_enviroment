import os
import cv2
import dlib
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift
import librosa
import soundfile as sf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def save_face(img,name, bbox, i, width=48,height=48):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    try:
        imgCrop = cv2.resize(imgCrop, (width, height))
        cv2.imwrite(os.path.join(name, i), imgCrop)
    except Exception as e:
        print(str(e) +"\n couldnt resize: "+ i)

def crop_faces(old_path, new_path, clear):
    detector = dlib.get_frontal_face_detector()

    if clear:
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(new_path+"Image/")):
            for f in filenames:
                os.remove(os.path.join(dirpath, f))

    for i, (root, dirs, files) in enumerate(os.walk(old_path)):
        for file in files:
            frame =cv2.imread(os.path.join(root, file))
            gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            dirpath_components = root.split("/")
            semantic_label = dirpath_components[-1]

            for face in faces:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                save_face(gray,os.path.join(new_path, semantic_label),(x1,y1,x2,y2), file)
            
            if not faces:
                print("No face found: "+ file)

def augment_audio_data(path, aug_path):
    augment = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)
    ])

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(aug_path+"Audio/")):
        for f in filenames:
            os.remove(os.path.join(dirpath, f))

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for f in filenames:
            signal, sr = librosa.load(os.path.join(dirpath, f))
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            # Save original
            label = semantic_label + "/" + f
            sf.write(os.path.join(aug_path, label), signal, sr)
            # Save 10 aug
            for count in range(0, 10):
                label = semantic_label + "/" + str(count) + "_" + f
                augmented_signal = augment(signal, sr)
                sf.write(os.path.join(aug_path, label), augmented_signal, sr)

def augment_image_data(path, aug_path):
    datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(aug_path+"Image/")):
        for f in filenames:
            os.remove(os.path.join(dirpath, f))
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for f in filenames:
            pic = load_img(os.path.join(dirpath, f))
            pic_array = img_to_array(pic)

            X = pic_array.reshape((1,) + pic_array.shape) 

            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
                
            f_s = f.split(".")
            
            for x, val in zip(datagen.flow(X, batch_size=2, save_to_dir=os.path.join(aug_path, semantic_label), save_prefix=f_s[0], save_format=f_s[1]),range(9)):     
                pass

#augment_image_data("Data/Train/Raw/Image", "Data/Train/Aug/")
#crop_faces('Data/Train/Aug/Image', 'Data/Train/Pre/', True)
#crop_faces('Data/Train/Raw/Image', 'Data/Train/Pre/', False)
#crop_faces('Data/Test/Raw/Image', 'Data/Test/Pre/', True)
#augment_audio_data("Data/Train/Raw/Audio", "Data/Train/Pre/")