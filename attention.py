import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tensorflow as tf

import os
os.environ['http_proxy'] = ""
os.environ['https_proxy'] = ""

path = './att_model'

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

def genDat(num_seqs, given_steps=19, pred_steps=21):
    x1 = []
    x2 = []
    y = []
    tot_steps = given_steps + pred_steps
    for i in range(num_seqs):
        Gb = np.random.normal(200, 10)
        s = np.random.normal(10, 0.5)
        c = np.random.normal(0.15, 0.01)
        a = np.random.normal(0.2, 0.01)
        z0 = [Gb, 0, 0]
        u = int(1+20*np.random.rand())  # 1 to 20
        hist = np.empty((tot_steps, 2))
        hist[0] = np.array([Gb, u])
        for t in range(tot_steps-1):
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
        x1.append(hist[:given_steps])
        x2.append(hist[given_steps:, 1:])
        y.append(hist[given_steps:, :1])
    x1 = np.array(x1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return x1, x2, y

##x1, x2, y = genDat(50000, pred_steps=41)
##pre_dataset = tf.data.Dataset.from_tensor_slices((x1, x2, y))

# custom attention model

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units, activation='tanh')
        self.V = tf.keras.layers.Dense(1)

    def call(self, values):
        # values shape == (batch_size, timesteps, hidden size)
        # weights shape == (batch_size, timesteps, 1)
        weights = self.V(self.W(values))

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class AttentionModel(tf.keras.Model):
    def __init__(self, rec_units, att_units, dense_units):
        super(AttentionModel, self).__init__()
        self.LSTM1 = tf.keras.layers.LSTM(rec_units,
                                          return_sequences=True,
                                          return_state=True)
        self.LSTM2 = tf.keras.layers.LSTM(rec_units,
                                          return_sequences=True)
        self.att = AttentionLayer(att_units)
        self.c_map = tf.keras.layers.Dense(rec_units, activation='tanh')
        self.dense = tf.keras.layers.Dense(dense_units, activation='relu')
        self.final = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x1, x2):
        x1, h, c = self.LSTM1(x1)
        new_h = self.att(x1)
        new_c = self.c_map(c)
        x2 = self.LSTM2(x2, initial_state=[new_h, new_c])
        return self.final(self.dense(x2))

model = AttentionModel(64, 64, 64)
model.load_weights(path)

# custom training

def loss(model, x1, x2, y):
    return tf.keras.losses.MSE(y, model(x1, x2))

def grad(model, input1, input2, target):
    with tf.GradientTape() as tape:
        loss_value = loss(model, input1, input2, target)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

##optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
##
##for epoch in range(1, 17):
##    dataset = pre_dataset.shuffle(1000).batch(50)
##    epoch_loss_avg = tf.keras.metrics.Mean()
##    print("Starting Epoch " + str(epoch))
##    for x1, x2, y in dataset:
##        loss_value, grads = grad(model, x1, x2, y)
##        optimizer.apply_gradients(zip(grads, model.trainable_variables))
##        epoch_loss_avg.update_state(loss_value)
##    print("Loss: " + str(epoch_loss_avg.result()))
##
##model.save_weights(path)

# test

x1_test, x2_test, y_test = genDat(1, pred_steps=81)

y_pred = model(x1_test, x2_test)

t = np.linspace(0, 100, 101)
G = [l[0] for l in x1_test[0]] + [l[0] for l in y_test[0]]
G_pred = np.concatenate((x1_test[0][-1][:1], y_pred.numpy().squeeze()))

plt.figure(figsize=(5, 5))
plt.subplot(1,1,1)
plt.plot(t[18:-1], G_pred, 'r')
plt.plot(t[:-1], G, 'g--')
plt.show()
