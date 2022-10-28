
import librosa
import librosa.display
import os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten


def getAudio(dir):
    res = os.listdir(dir)
    print(res)
def readFile(path):
    X =[]
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            # print(path+file)
            audio_data, sr = librosa.load(path+file, sr=8000, mono=True)
            melspectrum = librosa.feature.melspectrogram(y=audio_data, sr=sr, hop_length=512, window='hann', n_mels=256)
            r, c = melspectrum.shape
            if c==4 and r ==256:
                X.append(melspectrum.T)
    X = np.asarray(X)
    # print(X.shape)
    return X

DATASET_PATH_YES = 'traindata/bixby/'
DATASET_PATH_NO = 'traindata/nonbixby/'
nb_epoch = 10
samples_per_epoch = 1000
batch_size = 32
save_best_only = True
learning_rate = 1e-4
# data_dir = pathlib.Path(DATASET_PATH)
#
# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
# print('Commands:', commands)
X_yes = readFile(DATASET_PATH_YES)
Y_yes = np.zeros(X_yes.shape[0])+1
X_no = readFile(DATASET_PATH_NO)
Y_no = np.zeros(X_no.shape[0])

X = np.append(X_yes, X_no,0)
Y = np.append(Y_yes, Y_no,0)

pos = np.arange(Y.shape[0])
np.random.shuffle(pos)

x = X[pos,:,:]
y = Y[pos]
x = x.reshape(x.shape[0], 4, 256, 1)
# Chia ra traing set và validation set
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)
model = Sequential()
model.add(Conv2D(128,3, 3, activation='relu', input_shape=(4,256,1)))
model.add(Conv2D(64, 1, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# H = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
#           batch_size=32, epochs=10, verbose=1)

checkpoint = ModelCheckpoint('models/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=save_best_only,
                                 mode='auto')
H = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          batch_size=32, epochs=0, verbose=1)
# model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

