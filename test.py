import os
import glob
import numpy as np
import pandas as pd
from tensorflow import keras

def test(path="deploy", weight_path="model"):
    # Path to data
    test_path = os.path.join(path, "test")
    
    # Training setting
    num_classes = 3
    input_shape = [1052, 1914]

    model = keras.models.load_model(weight_path)
    # Build submission
    x_test = glob.glob(os.path.join(test_path, '*/*.jpg'))

    df_test = pd.DataFrame()
    df_test['guid/image'] = x_test
    df_test['label'] = ['0'] * len(x_test)

    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale= 1./255
    )
    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col = 'guid/image',
        y_col = 'label',
        target_size = input_shape,
        shuffle = False,
        batch_size = 1
    )

    # Predict
    y_test = model.predict(test_generator, batch_size=1)
    y_test = np.argmax(y_test, axis=1)

    # Save output
    df_test['label'] = y_test
    df_test['guid/image'] = df_test['guid/image'].apply(lambda x: x.replace('_image.jpg',''))
    df_test['guid/image'] = df_test['guid/image'].apply(lambda x: os.path.relpath(x, test_path))
    print(df_test)
    df_test.to_csv("submit.csv", index=False)

if __name__ == "__main__":
    test()