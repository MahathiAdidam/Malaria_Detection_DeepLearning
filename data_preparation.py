import os
import cv2
import numpy as np

def load_data(data_dir):
    images = []
    labels = []
    for label in ['Parasite', 'Uninfected']:
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(1 if label == 'Parasite' else 0)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    train_dir = 'D:/Machine_Learning/Malaria_Detection/Dataset/Train'
    test_dir = 'D:/Machine_Learning/Malaria_Detection/Dataset/Test'

    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    np.savez('data_split.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Data preparation complete: data_split.npz file created.")