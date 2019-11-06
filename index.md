## Ruuvi Fridge

This is my first project to collect data with [Ruuvi tag](https://ruuvi.com/) from my refrigerator and use that data to machine learning models. 

### Data
Temperature and humidity over night in my fridge. Blue lines mark where the fridge's motor starts and stops. Red line is when fridge door is opened.
![Image](https://github.com/kilkki/ruuvi-fridge/blob/master/graph1.png "Normal loop")

Motor starts (blue line)
* Temperature decreases
* Humidity decreases

Motor stops (blue line)
* Temperature and humidity starts to raise

Fridge door opens
* Sudden jump in humidity and temperature


Fridge door opened. Notice rapid raise of humidity and temperature
![Image](https://github.com/kilkki/ruuvi-fridge/blob/master/Screenshot_2019-11-01%20Ruuvi%20tagit%20-%20door_open.png "Door open")

### Predicting temperature using Tensor Flow and Recurrent Neural Network(RNN)
Predicting temperature in fridge when door is closed is easy but lets see how RNN can handle when fridge door is opened

 
```python
import tensorflow as tf
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import plotly.offline as py
import matplotlib.pyplot as plt
```

```python
# Data files
csv_path1 = "data/ruuvi_20191025.csv"
csv_path2 = "data/ruuvi_20191026.csv"
csv_path3 = "data/ruuvi_20191027.csv"
csv_path4 = "data/ruuvi_20191028.csv"
```

```python
# Load files
df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)
df3 = pd.read_csv(csv_path3)
df4 = pd.read_csv(csv_path4)
```

```python
# Combine datasets
df = pd.concat([df1,df2,df3,df4])
```

```python
# Filter by tag address
tag2_address = 'e7:99:d1:af:26:34'
df = df[df.address == tag2_address]
```

```python
# Convert (strange from InfluxDB) unix time to date object
df['date'] = pd.to_datetime(df['time'] / 1000000000,unit='s')
df.index = pd.to_datetime(df.date)
```

```python
# Combine acceleration absolute values to new feature
df['movement'] = df['accelerationX'].abs() + df['accelerationY'].abs() + df['accelerationZ'].abs()
```

```python
# Remove unused columns
features_considered = ['temperature', 'humidity', 'movement']
df = df[features_considered]
```

In dataset are samples less than second intervals. We don't need so many data points. Lets resample points to average of 20 second steps.
```python
# Downsample
df = df.resample('20S').mean() 
```
Variation in values of temperature and movement are hardly noticable. Data scaling needed.
![Image](https://github.com/kilkki/ruuvi-fridge/blob/master/data_unscaled.png "Unscaled data")

Lets scale data values between 0 and 1
```python
# Scale data
scaler_temperature= MinMaxScaler(feature_range=(0, 1))
data_1 = scaler_temperature.fit_transform(df.temperature.values.reshape(-1, 1))
scaler_humidity= MinMaxScaler(feature_range=(0, 1))
data_2 = scaler_temperature.fit_transform(df.humidity.values.reshape(-1, 1))
scaler_movement = MinMaxScaler(feature_range=(0, 1))
data_3 = scaler_movement.fit_transform(df.movement.values.reshape(-1, 1))

data = pd.DataFrame(data_1, columns=["temperature"])
data['humidity'] = pd.DataFrame(data_2)
data['movement'] = pd.DataFrame(data_3)

input_feature= data.iloc[:,[0,1,2]].values
input_data = input_feature
```
After scaling data is more readable and more efficient for neural network to learn weights.
![Image](https://github.com/kilkki/ruuvi-fridge/blob/master/data_scaled.png "Scaled data")
Now we can see spike in ruuvi tags accelemetor when fridge door has moved

```python
# function to break data, scale data, window data
def get_window_data(data, window):
    X = []
    y = []

    for i in range(len(data) - window - 1):
        X.append(data[i : i + window])
        y.append(data[i + window + 1])

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, scaler
```


```python
window_size = 100
X, y, scaler = get_window_data(input_data, window_size)
test_split = int(len(df) * 0.8)

X_train = X[:test_split]
X_test = X[test_split:]

y_train = y[:test_split]
y_test = y[test_split:]


X_train.shape , X_test.shape, y_train.shape, y_test.shape
```

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units= 50, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.LSTM(64,return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")
```

```python
history = model.fit(X_train, y_train, batch_size=256, nb_epoch=2, validation_data=(X_test, y_test))
```

```python
def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"],  label="Train")
    plt.plot(history.history["val_loss"], label="Test")
    plt.title("Loss over epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot(X_test):
    pred = model.predict(X_test)
    #pred_inverse = scaler.inverse_transform(pred.reshape(-1, 1))
    #y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate mean_squared_error. Previosly we did MinMax scale, so apply inverse_transform to recover values
    rmse = sqrt(mean_squared_error(y_test[:,0], pred))
    print('Test RMSE: %.3f' % rmse)
    plt.figure(figsize=(15, 8))
    plt.plot(pred, label='predict')
    plt.plot(y_test[:,0], label='actual')
    plt.legend()
    plt.show()
```
```python
plot_history(history)

plot(X_test)
```


