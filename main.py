
# Removed top-level references to model, x_test, le, history, and related variables that cause NameError

# --- REAL-TIME MONITORING FUNCTION ---
import cv2
def real_time_monitoring():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, (256, 256))
        img = img / 255.0
        pred = model.predict(img.reshape(1, 256, 256, 3))
        label = le.inverse_transform([pred.argmax()])[0]
        cv2.putText(frame, f'Engagement: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Engagement Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# To run real-time monitoring, uncomment the following line:
# real_time_monitoring()
import os
import cv2
import random 
import shutil
import warnings 
import numpy as np
import pandas as pd 
import seaborn as sns 
import tensorflow as tf
from tensorflow import keras
from tqdm.notebook import tqdm
import matplotlib.pyplot  as plt
warnings.filterwarnings("ignore")
from keras.optimizers import Adam
from tensorflow.keras import Model
from keras.models import Sequential 
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten , Dense , Conv2D, Dropout , MaxPooling2D 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense


# Updated dataset paths for local Windows environment
Dataset_path = r'D:\Project\Class_Monitoring\archive (7)'
Target_path = r'D:\Project\Class_Monitoring\Student-engagement-Dataset'

# create train/test/validation dirs
Train_dir = os.path.join(Target_path,'train')
valid_dir = os.path.join(Target_path,'validation')
Test_dir = os.path.join(Target_path,'test')

os.makedirs(Target_path,exist_ok = True)
os.makedirs(Train_dir,exist_ok=True)
os.makedirs(valid_dir,exist_ok=True)
os.makedirs(Test_dir,exist_ok=True)

img_size = (256,256)

image_paths = []
labels = []
preprocced_image_paths = [] 
preprocced_image_paths_test=[]   
for path in os.listdir(Dataset_path) :                                         #label = eng , diseng
        main_classes_dir = os.path.join(Dataset_path,path)                          #print(main_classes_dir)
        for main_path in os.listdir(main_classes_dir):                             #print labels=fursted,bored,drowsy,confused,..
            
            sub_classes_dir = os.path.join(main_classes_dir,main_path)
            
            img_count = len(os.listdir(sub_classes_dir)) 
            test_img_count = int(0.2*img_count)
            train_img_count = img_count - test_img_count
            
            target_train_dir = os.path.join(Train_dir,main_path)
            target_test_dir = os.path.join(Test_dir,main_path)
            target_validation_dir = os.path.join(valid_dir,main_path)
        
            os.makedirs(target_train_dir,exist_ok =True)                          #files with labels 
            os.makedirs(target_test_dir,exist_ok=True)
            os.makedirs(target_validation_dir,exist_ok=True)
            for sub_main_path in os.listdir(sub_classes_dir):
                
                  image_path = os.path.join(sub_classes_dir,sub_main_path)
                  image_paths.append(image_path)
                  labels.append(main_path)
                
                                                                                #Resize ,normlization 
                  img = cv2.imread(image_path)
                  img = cv2.resize(img,img_size)
#                   img = img / 255
            
                                                                                #Split train,test images 
                  
            
                  if len(os.listdir(target_test_dir)) != test_img_count:
                        cv2.imwrite(os.path.join(target_test_dir, sub_main_path), img)
                        preprocced_image_paths.append(os.path.join(target_test_dir,sub_main_path))
#                         preprocced_image_paths_test.append(os.path.join(target_test_dir,sub_main_path))
                  else:
                        cv2.imwrite(os.path.join(target_train_dir,sub_main_path),img )
                        preprocced_image_paths.append(os.path.join(target_train_dir,sub_main_path))
                      
print(image_paths[0:2])

img_count = len(image_paths)
test_img_count = int(0.2*img_count)
train_img_count = img_count - test_img_count

print("img_count",img_count)
print("test_no", test_img_count)
print("train_img_no",train_img_count)


dataset = pd.DataFrame()
dataset["label"],dataset["image"],dataset['preproccesd_image']= labels,image_paths,preprocced_image_paths
dataset = dataset.sample(frac=1).reset_index(drop=True)
print(dataset["image"][3])
print(dataset['preproccesd_image'][3])
dataset.head()


# print(min_shape)

def min_shape(image_paths) : 
    min_shape = [np.inf,np.inf,3]

# Removed obsolete and broken code blocks with indentation/undefined errors
batch_size = 32
img_size = (256, 256,3)

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
        Train_dir,
        batch_size=batch_size,
        class_mode = 'categorical',
        shuffle=True)



print(train_generator.class_indices)


def dataset_Loader(dir_path) : 
    image_paths = []
    labels = []
    for label in os.listdir(dir_path) :
        files = os.path.join(dir_path,label)
        for filename in os.listdir(files) :
            image_path = os.path.join(dir_path,label,filename)
            image_paths.append(image_path)
            labels.append(label)
        print(label,"Completed")
        
    return image_paths,labels
train = pd.DataFrame()

train["image"], train["label"] = dataset_Loader(Train_dir)

# --- ENGAGEMENT LABEL MAPPING ---
engaged_labels = ["engaged", "interested", "focused"]  # adjust as per your dataset
def map_engagement(label):
    if label.lower() in engaged_labels:
        return "engaged"
    else:
        return "disengaged"
train["engagement"] = train["label"].apply(map_engagement)


test = pd.DataFrame()

test["image"], test["label"] = dataset_Loader(Test_dir)
test["engagement"] = test["label"].apply(map_engagement)


train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)


def extract_feature(images) :
    features = []
    for image in tqdm(images) :
        img = load_img(image) 
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 256, 256, 3)
    return features
target_test_dir = os.path.join(Target_path+'1',"test")
test_ = pd.DataFrame()
test_["preprocced_image"],test_["label"] = preprocceed_test_images,test_labels
test_ = test_.sample(frac=1).reset_index(drop=True)
x_test_ = extract_feature(test_['preprocced_image'])
x_test_ = x_test_
pred = model.predict(x_test_[image_index].reshape(1,256,256,3))
prediction_label = le.inverse_transform([pred.argmax()])[0]
pred = model.predict(x_test_[image_index].reshape(1,256,256,3))
prediction_label = le.inverse_transform([pred.argmax()])[0]
pred = model.predict(x_test_[image_index].reshape(1,256,256,3))
prediction_label = le.inverse_transform([pred.argmax()])[0]
pred = model.predict(x_test_[image_index].reshape(1,256,256,3))
prediction_label = le.inverse_transform([pred.argmax()])[0]

# --- MODEL TRAINING, EVALUATION, AND METRICS ---
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

train_features = extract_feature(train["image"])
test_features = extract_feature(test["image"])
x_train = train_features / 255.0
x_test = test_features / 255.0

le = LabelEncoder()
le.fit(["engaged", "disengaged"])
y_train = le.transform(train["engagement"])
y_test = le.transform(test["engagement"])
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

img_size = (256, 256, 3)
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=img_size),
    keras.layers.MaxPooling2D(3, 3),
    keras.layers.Conv2D(64, (1, 1), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(3, 3),
    keras.layers.Conv2D(128, (1, 1), activation='relu'),
    keras.layers.Conv2D(128, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D(3, 3),
    keras.layers.Conv2D(256, (1, 1), activation='relu'),
    keras.layers.Conv2D(256, (5, 5), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.summary()

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

checkpoint = ModelCheckpoint("Student_Engagement_Model.h5", monitor='val_accuracy', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=9, verbose=1, mode='auto')

history = model.fit(x=x_train, y=y_train, epochs=30, callbacks=[checkpoint, early_stopping])

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(x_test)
y_pred_labels = le.inverse_transform(y_pred.argmax(axis=1))
y_true_labels = le.inverse_transform(y_test.argmax(axis=1))
print(classification_report(y_true_labels, y_pred_labels))
print(confusion_matrix(y_true_labels, y_pred_labels))

plt.plot(history.history["accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy"])
plt.show()

plt.plot(history.history['loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss"])
plt.show()

print("Script completed successfully.")
