import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def create_model(num_classes):
    model = keras.Sequential([
        keras.Input(shape=[1052, 1914, 3]),
        layers.Conv2D(64, kernel_size=(3, 3), activation="sigmoid"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="sigmoid"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="sigmoid"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="sigmoid"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="sigmoid"),
        layers.Dense(64, activation="sigmoid"),
        layers.Dense(64, activation="sigmoid"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

def train(path="deploy", weight_path="model"):
    # Path to data
    train_path = os.path.join(path, "trainval")
    
    # Training setting
    epochs = 5
    num_classes = 3
    input_shape = [1052, 1914]
    batch_size = 4

    # Data loading
    y_train = pd.read_csv(os.path.join(path, 'trainval/labels.csv'))
    y_train = y_train.sort_values(by='guid/image')
    y_train['guid/image'] = y_train['guid/image'].apply(lambda x: os.path.join(train_path, x+'_image.jpg')) 
    y_train['label'] = y_train['label'].astype('str')
    print(y_train)

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale= 1./255
    )

    train_generator = train_datagen.flow_from_dataframe(
        y_train, 
        x_col='guid/image', 
        y_col = 'label',
        target_size = input_shape,
        class_mode = 'categorical',
        batch_size = batch_size
    )

    # Model
    model = create_model(num_classes)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_generator, batch_size=batch_size, epochs=epochs)
    # Save the whole model
    model.save(weight_path)

if __name__ == "__main__":
    train()