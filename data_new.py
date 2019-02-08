# Copyright 2018 Google, Inc.,
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset for iterating over MIDI files."""

from __future__ import print_function

import numpy as np
import pretty_midi
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import tensorflow.contrib.rnn as rnn

batch_s = 64
def pitch_to_onehot(pitch):
    pitch[pitch!=0]=1
    location = None
    for j in range(0,len(pitch)):
        if pitch[j] == 1:
            if location == None:
                location = j
            else:
                pitch[location] = 0
                location = j
    t = max(pitch)
    # pitch = np.append(pitch,0)
    # if max(pitch) == 0:
    #     pitch[128]=1.0
    #print(t,max(pitch))
    return pitch

def calculate_proper_rate(beats):
    bmp = beats[1][0]
    time_per_beat = 60/bmp
    beats_per_bar = 4
    amount = time_per_beat*beats_per_bar
    rate=(amount/16)*100
    return rate

def plot_sequences(roll):
    # 'nearest' interpolation - faithful but blocky
    plt.imshow(tf.transpose(roll),aspect='auto')
    plt.ylim([0,128])
    plt.show()

def piano_roll_sequences(filenames, batch_size, sequence_size, rate=10):
    """Returns a dataset of piano roll sequences from the given files.."""

    def _to_piano_roll(filename, sequence_size):
        """Load a file and return consecutive piano roll sequences."""
        try:
            midi = pretty_midi.PrettyMIDI(tf.compat.as_text(filename))
        except Exception:
            print("Skipping corrupt MIDI file", filename)
            return np.zeros([0, sequence_size, 128], dtype=np.float64)
        if midi.time_signature_changes != []:
            if midi.time_signature_changes[0].numerator == 4 & midi.time_signature_changes[0].denominator == 4:
                rate = calculate_proper_rate(midi.get_tempo_changes())
                midi_get = midi.get_piano_roll(rate)
                roll = np.asarray(midi_get.transpose(), dtype=np.float64)
                roll = np.asarray([pitch_to_onehot(val) for val in roll])

                #plot_sequences(roll.transpose())
                assert roll.shape[1] == 128
                # Pad the roll to a multiple of sequence_size
                length = len(roll)
                remainder = length % sequence_size
                if remainder:
                    new_length = length + sequence_size - remainder
                    roll = np.resize(roll, (new_length, 128))
                    roll[length:, :] = False
                    length = new_length
                return np.reshape(roll, (length // sequence_size, sequence_size, 128))
            return np.zeros([0, sequence_size, 128], dtype=np.float64)

    def _to_piano_roll_dataset(filename):
        """Filename (string scalar) -> Dataset of piano roll sequences."""
        sequences, = tf.py_func(_to_piano_roll,
                                [filename, sequence_size],
                                [tf.float64])
        #sequences.set_shape([None, None, 128])
        sequences.set_shape([sequences.shape[0],sequences.shape[1], 128])
        roll = tf.data.Dataset.from_tensor_slices(sequences)
        return roll

    batch_size = tf.to_int64(batch_size)
    return (tf.data.Dataset.from_tensor_slices(filenames)
            .interleave(_to_piano_roll_dataset,
                        cycle_length=batch_size * 5,
                        block_length=1)
            .repeat()
            .batch(batch_size))

def piano_roll_to_midi(piano_roll, sample_rate, filename):
    """Convert the piano roll to a PrettyMIDI object.
    See: http://github.com/craffel/examples/reverse_pianoroll.py
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    midi.instruments.append(instrument)
    padded_roll = np.pad(piano_roll, [(1, 1), (0, 0)], mode='constant')
    changes = np.diff(padded_roll, axis=0)
    notes = np.full(piano_roll.shape[1], -1, dtype=np.int)
    for tick, pitch in zip(*np.where(changes)):
        prev = notes[pitch]
        if prev == -1:
            notes[pitch] = tick
            continue
        notes[pitch] = -1
        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=prev / float(sample_rate),
            end=tick / float(sample_rate)))
    midi.write("./output/"+filename)
    return midi


def write_test_note(path, duration, note):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, note, 0.0, duration))
    midi.instruments.append(instrument)
    midi.write(path)

def eager(filenames, batch_size, sequence_size):
    tf.enable_eager_execution()

    """Returns a dataset of piano roll sequences from the given files.."""

    def _to_piano_roll(filename, sequence_size):
        """Load a file and return consecutive piano roll sequences."""
        try:
            midi = pretty_midi.PrettyMIDI(tf.compat.as_text(filename))
        except Exception:
            print("Skipping corrupt MIDI file", filename)
            return np.zeros([64, sequence_size, 128], dtype=np.float64)
        if midi.time_signature_changes != []:
            if midi.time_signature_changes[0].numerator == 4 & midi.time_signature_changes[0].denominator == 4:
                rate = calculate_proper_rate(midi.get_tempo_changes())
                midi_get = midi.get_piano_roll(rate)
                roll = np.asarray(midi_get.transpose(), dtype=np.float64)
                roll = np.asarray([pitch_to_onehot(val) for val in roll])
                #piano_roll_to_midi(roll,rate,"bachtest.mid")
                #plot_sequences(roll)
                assert roll.shape[1] == 128
                # Pad the roll to a multiple of sequence_size
                length = len(roll)
                remainder = length % sequence_size
                if remainder:
                    new_length = length + sequence_size - remainder
                    roll = np.resize(roll, (new_length, 128))
                    roll[length:, :] = False
                    length = new_length
                #batch_s = length//sequence_size
                return np.reshape(roll, (length // sequence_size, sequence_size, 128))
            return np.zeros([64, sequence_size, 128], dtype=np.float64)

    def _to_piano_roll_dataset(filename):
        """Filename (string scalar) -> Dataset of piano roll sequences."""
        sequences = _to_piano_roll(filename,sequence_size)
        # sequences.set_shape([None, None, 128])
        sequences.shape = (sequences.shape[0], sequences.shape[1], 128)

        roll = tf.convert_to_tensor(sequences)
        return roll

    batch_size = tf.to_int64(batch_size)
    full = []
    for filename in filenames:
        dataset = _to_piano_roll_dataset(filename)
        full.append(dataset[0:64])
        #full.append(dataset[33:65])
    return full

def init_data(batch_size,sequence_size):
    files = glob.glob(".\\midi\\bach\\test" + '/**/*.mid', recursive=True)
    print(files)
    data = eager(files, batch_size, sequence_size)
    print("DATA PROCESSING COMPLETE")
    return data

def init_model(checkpoint_dir, learning_rate, restore, single_model):
    model = model_layers(single_model)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    #checkpoint_dir = './log/model/good_smallbatch/'
    if restore:
        root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return model,optimizer,root

def eager_autoregressive(checkpoint_dir):
    learning_rate = 0.001
    num_steps = 1000
    batch_size = 64
    sequence_size = 64
    display_step = 20

    data = init_data(batch_size, sequence_size)
    model, optimizer, root = init_model(checkpoint_dir, learning_rate, True)
    train_writer = tf.contrib.summary.create_file_writer('./log/good_smallbatch', flush_millis=1000)
    for epoch in range(num_steps):
        for index in range(0, len(data)):
            with tf.GradientTape() as tape:
                current_pred, new_states = tf.nn.dynamic_rnn(model, data[index], dtype=tf.float64)
                # # Prediction
                y_pred = current_pred
                # # Targets (Labels) are the input data.
                y_true = data[index]
                # # Define loss and optimizer, minimize the squared error
                loss = tf.losses.softmax_cross_entropy(y_true, y_pred)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                with train_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("loss", loss, step=epoch)
                print(epoch, loss)
        if epoch % display_step == 0:
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            root.save(checkpoint_prefix)


def eager_autoencoder(checkpoint_dir):
    learning_rate = 0.001
    num_steps = 1000
    batch_size = 64
    sequence_size = 64
    display_step = 20

    data = init_data(batch_size,sequence_size)
    model,optimizer,root = init_model(checkpoint_dir,learning_rate,True)
    train_writer = tf.contrib.summary.create_file_writer('./log/good_smallbatch', flush_millis=1000)
    for epoch in range(num_steps):
        for index in range(0,len(data)):
            with tf.GradientTape() as tape:
                current_pred, new_states = tf.nn.dynamic_rnn(model, data[index], dtype=tf.float64)
                # # Prediction
                y_pred = current_pred
                # # Targets (Labels) are the input data.
                y_true = data[index]
                # # Define loss and optimizer, minimize the squared error
                loss = tf.losses.softmax_cross_entropy(y_true, y_pred)
                grads = tape.gradient(loss,model.trainable_variables)
                optimizer.apply_gradients(zip(grads,model.trainable_variables))
                with train_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("loss", loss, step=epoch)
                print(epoch,loss)
        if epoch % display_step == 0:
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            root.save(checkpoint_prefix)

def save_prediction_to_file(y_pred):
    y_pred = tf.nn.softmax(y_pred)
    matrix = tf.reshape(y_pred, [y_pred.shape[0] * y_pred.shape[1], 128])
    np.savetxt("test.txt", matrix.numpy())

def model_layers(single):
    if single:
        return tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [128,32,128]])
    else:
        return [rnn.MultiRNNCell([lstm_cell(size) for size in [128,32]]),rnn.MultiRNNCell([lstm_cell(size) for size in [32,128]])]

def lstm_cell(state_value):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_value, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    return cell

def multiply(z_data):
    z_data = z_data.numpy()
    for i,batch in enumerate(z_data):
        for j,seq in enumerate(batch):
            z_data[i,j] = multiply(seq)
    t1 = np.linspace(1,0,16,endpoint=False)
    t2 = np.linspace(0,1,16)
    t3 = np.append(t1,t2)
    test = tf.math.multiply(z_data,t3)
    return test

def load_model():
    c = np.loadtxt("test.txt")
    # piano_roll_to_midi(Y,20)
    newarray = []
    for index,val in enumerate(c):
        if max(val) > -10:
            newarray.append(tf.argmax(val))
        else:
            newarray.append(-1)
    newarray = tf.convert_to_tensor(newarray)
    Y = tf.one_hot(tf.convert_to_tensor(newarray),depth=128)
    t = Y[0]
    #Y = tf.one_hot(tf.argmax(c, dimension=1), depth=128)
    plot_sequences(Y)
    #Y = Y[..., :-1]
    #Y = np.asarray([fgt[0:-1] for fgt in Y])
    piano_roll_to_midi(Y,10,"bach_compare_goodsall10.mid")


tf.enable_eager_execution()
#autoencoder()
eager_autoencoder()
#load_model()