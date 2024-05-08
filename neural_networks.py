import os

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense


def get_best_neural_network(X_test, y_test, nn_path):
    files = os.listdir(nn_path)
    files.sort()
    best_weights_file = os.path.join(nn_path, files[-1])  # Latest checkpoint performed best.
    nn_model = create_seq_nn(X_test.shape[1])
    nn_model.load_weights(best_weights_file)  # load checkpoint
    return nn_model


def train_neural_network(X_train, y_train, nn_path):
    number_columns = X_train.shape[1]
    nn_model = create_seq_nn(number_columns)

    if not os.path.exists(nn_path):
        os.makedirs(nn_path)
    else:
        try:
            files = os.listdir(nn_path)
            for file in files:
                file_path = os.path.join(nn_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Old network checkpoints deleted successfully.")
        except OSError:
            print("Error occurred while deleting old checkpoint files.")

    checkpoint_name = os.path.join(nn_path, 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks_list)


def create_seq_nn(input_dim):
    """
    Creates a simple sequential neural network for regression (1 linear output layer), with three hidden layers.
    :param input_dim: Number of variables in the input data.
    :return: the neural network.
    """
    nn = Sequential()

    # The Input Layer:
    nn.add(Dense(32, kernel_initializer='normal', input_dim=input_dim, activation='relu'))

    # Three Hidden Layers:
    nn.add(Dense(64, kernel_initializer='normal', activation='relu'))
    nn.add(Dense(64, kernel_initializer='normal', activation='relu'))
    nn.add(Dense(64, kernel_initializer='normal', activation='relu'))

    # The Output Layer:
    nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile the network:
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # nn.summary()
    return nn