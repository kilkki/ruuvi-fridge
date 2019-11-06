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
