import os
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.ticker as ticker
from keras.utils.vis_utils import plot_model
import datetime

# hyperparameter
lookback = 1
step = 1
delay = 1
batch_size = 1
train_size = 600
val_size = 110
# test_size = 365 Meet one year
predict_num_day = 365+20
# site_list = ['T0100','T0900','T1700']
site_list = ['T0100']

# Define generator
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=8, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][-1]
        yield samples, targets


# Define model
def create_model(dataset_shape):
    model = Sequential()
    model.add(layers.LSTM(32,
                         # dropout=0.1,
                         # recurrent_dropout=0.1,
                         return_sequences=True,
                         input_shape=(None, dataset_shape[-1])))
    # model.add(layers.LSTM(64, activation='relu',
    #                      # dropout=0.1,
    #                      # recurrent_dropout=0.1,
    #                      return_sequences=True))
    model.add(layers.LSTM(32, activation='relu',
                         # dropout=0.1,
                         # recurrent_dropout=0.1
                         ))
    model.add(layers.Dense(1))
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# Define callback function
def create_callbacks(opt):
    callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto'),
    ModelCheckpoint('optimizers_best_' + opt + '.h5', monitor='val_loss', save_best_only=True, verbose=0)
    ]
    return callbacks

# create a data set
def create_dataset(site):
    # Read data
    dataframe = read_csv('/home/lk/Documents/AI/mathmodel/Datasets/final_data.csv',
                         usecols=[site])
    data_set = dataframe.values

    # Integer to float
    data_set = data_set.astype('float32')

    # Data processing, normalized to between 0 ~ 1
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_set[:train_size])
    train_data = scaler.transform(data_set[:train_size])
    test_data = scaler.transform(data_set[train_size:])
    dataset = np.vstack((train_data,test_data))

    dataset_shape = dataset.shape

    # Divide the data set
    train_gen = generator(dataset,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=train_size,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(dataset,
                        lookback=lookback,
                        delay=delay,
                        min_index=train_size+1,
                        max_index=train_size+1+val_size,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(dataset,
                         lookback=lookback,
                         delay=delay,
                         min_index=train_size+1+val_size+1,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # val_steps = (val_size - lookback) // batch_size
    # val_steps = Days you want to predict + Test set size
    val_steps = predict_num_day + (len(dataset)-train_size-val_size)

    # This is how many steps to draw from `test_gen`
    # test_steps = (len(dataset) - (train_size+1+val_size+1) - lookback) // batch_size
    test_steps = predict_num_day + (len(dataset)-train_size-val_size)
    return train_gen, val_gen, test_gen, val_steps, test_steps, dataset_shape, scaler, data_set



def main_function(site):
    train_gen, val_gen, test_gen, val_steps, test_steps, dataset_shape, scaler, data_set = create_dataset(site)
    model = create_model(dataset_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')
    callbacks = create_callbacks('adam')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=64,
                                  epochs=200,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, label='Training loss',linewidth=3.0, ms=10)
    plt.plot(epochs, val_loss, label='Validation loss',linewidth=3.0, ms=10)
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('loss value', fontsize=10)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.title('Training and validation loss',fontsize=20)
    plt.legend()
    if not os.path.exists('picture'):
        os.mkdir('picture')
    plt.savefig('./picture/'+ site + 'Training and validation loss' + '.png', dpi=100)
    plt.close('all')
    #plt.show()

def generate_date():
    # Generate date coordinates
    time = []
    read_time = read_csv('/home/lk/Documents/AI/mathmodel/Datasets/final_data.csv',
                         usecols=['date'])
    start_time = str(read_time.values[train_size+val_size][0])
    end_time = str(read_time.values[-1][0])
    start_time = str(start_time[0:4])+'-'+str(start_time[4:6])+'-'+str(start_time[-2:])
    end_time = str(end_time[0:4])+'-'+str(end_time[4:6])+'-'+str(end_time[-2:])
    datestart = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    dateend = datetime.datetime.strptime(end_time, '%Y-%m-%d')
    dateend += datetime.timedelta(days=predict_num_day)
    while datestart <= dateend:
        time.append(datestart.strftime('%Y-%m-%d'))
        datestart += datetime.timedelta(days=1)
    return time


def plot_function(site):
    train_gen, val_gen, test_gen, val_steps, test_steps, dataset_shape, scaler, data_set = create_dataset(site)
    best_model = create_model(dataset_shape)

    # Load the model weights with the highest validation accuracy
    best_model.load_weights('optimizers_best_adam.h5')
    best_model.compile(optimizer='adam', loss='mean_squared_error')

    # Test set prediction result
    preds_test = best_model.predict_generator(
        test_gen,
        test_steps
    )


    preds_test = scaler.inverse_transform(preds_test)
    next_year = preds_test[len(data_set)-train_size-val_size:]
    # print(next_year.shape)
    np.savetxt(site+'.csv', next_year, delimiter=',')

    # Evaluation error
    train_error = best_model.evaluate_generator(train_gen, val_steps)
    val_error = best_model.evaluate_generator(val_gen, val_steps)
    test_error = best_model.evaluate_generator(test_gen, test_steps)
    print("训练集的均方误差为：",train_error)
    print("验证集的均方误差为：",val_error)
    print("测试集的均方误差为：",test_error)


    # Drawing a picture
    time = generate_date()
    tick_spacing = 187.3
    fig, ax = plt.subplots(1, 1)
    ax.plot(time[0:len(data_set)-train_size-val_size], data_set[train_size+val_size:], label='Act',linewidth=3.0, ms=10)
    ax.plot(time, preds_test, label='Pre', linewidth=3.0, ms=10)
    plt.xlabel('time', fontsize = 10)
    plt.ylabel('value', fontsize = 10)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.title('Actual and predicted values', fontsize = 20)
    plt.legend()
    # plt.show()
    #save figure
    plt.savefig('./picture/'+ site +'Actual and predicted values' + '.png', dpi=100)
    plt.close('all')

if __name__ == '__main__':
   for site in site_list:
       main_function(site)
       plot_function(site)
