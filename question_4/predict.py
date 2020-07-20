'''
Created on 2019年2月16日
    时间序列预测问题可以通过滑动窗口法转换为监督学习问题
@author: Administrator
'''

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model


# 创建数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


if __name__ == '__main__':
    # 加载数据
    dataframe = read_csv('/home/lk/Documents/AI/mathmodel/Datasets/final_data.csv', usecols=['T0100'])
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    print(dataset.shape)

    # 数据处理，归一化至0~1之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # 创建测试集和训练集
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)  # 单步预测
    testX, testY = create_dataset(test, look_back)

    # 调整输入数据的格式
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))  # （样本个数，1，输入的维度）
    testX = numpy.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))

    # 创建LSTM神经网络模型
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # 绘制网络结构
    plot_model(model, to_file='model.png', show_shapes=True)

    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # 计算得分
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # 绘图
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()