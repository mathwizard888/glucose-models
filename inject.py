import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tensorflow as tf

import os
os.environ['http_proxy'] = ""
os.environ['https_proxy'] = ""

simple_path = './simple_model'
gru_path = './gru_model'
lstm_path = './lstm_model'

def diff(z,t,u,Gb,c,a,s):
    G=z[0]
    I=z[1]
    Ia=z[2]
    dGdt = Gb-G-s*I
    dIdt = -c*I+a*Ia
    dIadt = -a*Ia+u
    dzdt = [dGdt, dIdt, dIadt]
    return dzdt

# generate training data

def genDat(num_seqs, extend=0):
    x = []
    y = []
    for i in range(num_seqs):
        Gb = np.random.normal(200, 10)
        s = np.random.normal(10, 0.5)
        c = np.random.normal(0.15, 0.01)
        a = np.random.normal(0.2, 0.01)
        z0 = [Gb, 0, 0]
        u = int(1+20*np.random.rand())  # 1 to 20
        hist = np.empty((41, 2))
        hist[0] = np.array([Gb, u])
        for t in range(40):
            z = odeint(diff, z0, [t,t+1], args=(u,Gb,c,a,s))
            z0 = z[1]
            z0[0] = max(z0[0], 0)
            z0[1] = max(z0[1], 0)
            z0[2] = max(z0[2], 0)
            r = np.random.rand()
            if r < 0.2 and z0[0] > 50:
                u = int(1+100*r)
            else:
                u = 0
            hist[t+1] = np.array([z0[0], u])
        # normalize to [0,1]
        maxG = max([l[0] for l in hist])
        minG = min([l[0] for l in hist])
        maxU = max([l[1] for l in hist])
        for j in range(len(hist)):
            hist[j][0] = (hist[j][0]-minG) / (maxG-minG)
            hist[j][1] /= maxU
        for j in range(len(hist)-20-extend):
            x.append(hist[j:j+19])
            y.append(hist[j+1:j+20+extend])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return x, y

##x_train, y_train = genDat(5000, 10)
##pre_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# model with custom loss

def loss(y_true, y_pred):
    G_true = tf.split(y_true, 2, axis=-1)[0]
    G_pred = tf.split(y_pred, 2, axis=-1)[0]
    return tf.keras.losses.MSE(G_true, G_pred)

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True,
                         stateful=True, batch_input_shape=[50, None, 2]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])

lstm_model.load_weights(lstm_path)

# custom training

def future_loss(model, x, y):
    extension = tf.shape(y)[1] - tf.shape(x)[1]
    for t in range(extension+1):
        if t > 0:
            inp = tf.concat((pred[:,:,:1], y[:,t+17:t+18,1:]), -1)
        else:
            inp = x
        pred = model(inp)
        if t == 0:
            y_ = pred[:,:,:]
            pred = pred[:,-1:,:]
        else:
            y_ = tf.concat((y_, pred), 1)
    return loss(y, y_)

def grad(model, inputs, targets):
    model.reset_states()
    with tf.GradientTape() as tape:
        loss_value = future_loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

##optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
##
##for epoch in range(1, 17):
##    dataset = pre_dataset.shuffle(1000).batch(50)
##    epoch_loss_avg = tf.keras.metrics.Mean()
##    print("Starting Epoch " + str(epoch))
##    for x, y in dataset:
##        loss_value, grads = grad(lstm_model, x, y)
##        optimizer.apply_gradients(zip(grads, lstm_model.trainable_variables))
##        epoch_loss_avg.update_state(loss_value)
##    print("Loss: " + str(epoch_loss_avg.result()))
##    
##lstm_model.save_weights(lstm_path)

# test

simple_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, return_sequences=True,
                              stateful=True, batch_input_shape=[1, None, 2]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])
simple_model.load_weights(simple_path)
simple_model.build(tf.TensorShape([1, None, 2]))

gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True,
                        stateful=True, batch_input_shape=[1, None, 2]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])
gru_model.load_weights(gru_path)
gru_model.build(tf.TensorShape([1, None, 2]))

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True,
                         stateful=True, batch_input_shape=[1, None, 2]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])
lstm_model.load_weights(lstm_path)
lstm_model.build(tf.TensorShape([1, None, 2]))

x_test, y_test = genDat(1)
s_y_pred = []
g_y_pred = []
l_y_pred = []
s_inp = x_test[0]
g_inp = x_test[0]
l_inp = x_test[0]
simple_model.reset_states()
gru_model.reset_states()
lstm_model.reset_states()
for t in range(21):
    if t > 0:
        s_inp = tf.expand_dims(np.array([s_pred_G, x_test[t][-1][1]]), 0)
        g_inp = tf.expand_dims(np.array([g_pred_G, x_test[t][-1][1]]), 0)
        l_inp = tf.expand_dims(np.array([l_pred_G, x_test[t][-1][1]]), 0)
    s_inp = tf.expand_dims(s_inp, 0)
    g_inp = tf.expand_dims(g_inp, 0)
    l_inp = tf.expand_dims(l_inp, 0)
    s_pred = simple_model(s_inp)
    g_pred = gru_model(g_inp)
    l_pred = lstm_model(l_inp)
    if t == 0:
        s_pred_G = tf.squeeze(s_pred.numpy())[-1][0]
        g_pred_G = tf.squeeze(g_pred.numpy())[-1][0]
        l_pred_G = tf.squeeze(l_pred.numpy())[-1][0]
    else:
        s_pred_G = tf.squeeze(s_pred.numpy())[0]
        g_pred_G = tf.squeeze(g_pred.numpy())[0]
        l_pred_G = tf.squeeze(l_pred.numpy())[0]
    s_y_pred.append(s_pred_G)
    g_y_pred.append(g_pred_G)
    l_y_pred.append(l_pred_G)

t = np.linspace(0, 40, 41)
G = [l[0] for l in x_test[0]] + [l[-1][0] for l in y_test]
s_G_pred = [x_test[0][-1][0]] + s_y_pred
g_G_pred = [x_test[0][-1][0]] + g_y_pred
l_G_pred = [x_test[0][-1][0]] + l_y_pred

plt.figure(figsize=(5, 5))
plt.subplot(1,1,1)
plt.plot(t[18:-1], s_G_pred, 'r')
plt.plot(t[18:-1], g_G_pred, 'b')
plt.plot(t[18:-1], l_G_pred, 'k')
plt.plot(t[:-1], G, 'g--')
plt.show()
