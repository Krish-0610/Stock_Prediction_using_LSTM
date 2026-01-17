import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(x, y, time_step):
    xs, ys = [], []
    for i in range(time_step, len(x)):
        xs.append(x[i-time_step:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

def prepare_data(df, features, target_cols, time_step, split_ratio=0.7):
    split = int(len(df) * split_ratio)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    input_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler = MinMaxScaler(feature_range=(0,1))

    input_scaler.fit(train_df[features])
    target_scaler.fit(train_df[target_cols])

    train_x = input_scaler.transform(train_df[features])
    test_x = input_scaler.transform(test_df[features])

    train_y = target_scaler.transform(train_df[target_cols])
    test_y  = target_scaler.transform(test_df[target_cols])

    x_train, y_train = create_sequences(train_x, train_y, time_step)

    test_input_x = np.concatenate((train_x[-time_step:], test_x))
    test_input_y = np.concatenate((train_y[-time_step:], test_y))

    x_test, y_test = create_sequences(test_input_x, test_input_y, time_step)

    return x_train, y_train, x_test, y_test, target_scaler