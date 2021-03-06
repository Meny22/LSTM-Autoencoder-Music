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
    plt.xlim([0,100])
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

def init_encoder(checkpoint_dir, learning_rate, restore):
    model = model_layers(True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    #checkpoint_dir = './log/model/good_smallbatch/'
    if restore:
        root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return model,optimizer,root

def init_decoder(checkpoint_dir,learning_rate,restore):
    model = model_layers(False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    # checkpoint_dir = './log/model/good_smallbatch/'
    if restore:
        root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return model, optimizer, root

def eager_autoregressive(checkpoint_dir):
    learning_rate = 0.001
    num_steps = 10000
    batch_size = 64
    sequence_size = 64
    display_step = 20

    data = init_data(batch_size, sequence_size)
    encoder_model, encoder_optimizer, encoder_root = init_encoder(checkpoint_dir, learning_rate, False)
    decoder_model, decoder_optimizer, decoder_root = init_decoder(checkpoint_dir, learning_rate, False)
    train_writer = tf.contrib.summary.create_file_writer('./log/autoregressive_sectest', flush_millis=1000)
    for epoch in range(num_steps):
        for index in range(0, len(data)):
            with tf.GradientTape(persistent=True) as tape:
                #encoder
                z_output, new_states = tf.nn.dynamic_rnn(encoder_model, data[index], dtype=tf.float64)
                #enc_tape.watch(encoder_model.trainable_variables)
                #concatenation
                data_rel = data[index]
                matrix = tf.reshape(data_rel, [data_rel.shape[0] * data_rel.shape[1], 128])
                shifted_output = tf.roll(matrix, 1,axis=0)
                shifted_output = tf.concat([tf.zeros([1,128],dtype=tf.float64),shifted_output[1:]],0)
                fhgf = shifted_output[0]
                z_output = tf.reshape(z_output,(z_output.shape[0]*z_output.shape[1],32))
                decoder_input = tf.concat([shifted_output, z_output], 1)
                decoder_input = tf.reshape(decoder_input,(data_rel.shape[0],data_rel.shape[1],160))

                #decoder
                current_prediction, new_states = tf.nn.dynamic_rnn(decoder_model,decoder_input,dtype=tf.float64)
                #dec_tape.watch(decoder_model.trainable_variables)
                # # Prediction
                y_pred = current_prediction
                # # Targets (Labels) are the input data.
                y_true = data[index]
                # # Define loss and optimizer, minimize the squared error
                loss = tf.losses.softmax_cross_entropy(y_true, y_pred)
            grads_encoder = tape.gradient(loss, encoder_model.trainable_variables)
            encoder_optimizer.apply_gradients(zip(grads_encoder, encoder_model.trainable_variables))
            grads_decoder = tape.gradient(loss, decoder_model.trainable_variables)
            decoder_optimizer.apply_gradients(zip(grads_decoder, decoder_model.trainable_variables))
            with train_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("loss", loss, step=epoch)
            print(epoch, loss)
        if epoch % display_step == 0:
            enc_checkpoint_prefix = os.path.join(checkpoint_dir+"/enc", "ckpt")
            dec_checkpoint_prefix = os.path.join(checkpoint_dir+"/dec", "ckpt")
            encoder_root.save(enc_checkpoint_prefix)
            decoder_root.save(dec_checkpoint_prefix)


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

def model_layers(encoder):
    if encoder:
        return tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [128,32]])
    else:
        return tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [160,128]])

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

def concatenate(z_vector, data):
    z_vector = z_vector.numpy()
    concatenated = np.zeros((64,64,160))
    for i, batch in enumerate(data):
        for j, seq in enumerate(batch):
            concatenated[i,j] = tf.concat((seq,z_vector[i,j]))
    return concatenated

def generate(batch_size,sequence_size):
    checkpoint_dir = "./log/model/autoregressive_sectest"
    learning_rate = 0.001
    encoder_model, encoder_optimizer, encoder_root = init_encoder(checkpoint_dir+"/enc", learning_rate, True)
    decoder_model, decoder_optimizer, decoder_root = init_decoder(checkpoint_dir+"/dec", learning_rate, True)
    data = init_data(batch_size,sequence_size)
    z_output, states = tf.nn.dynamic_rnn(encoder_model,data[0],dtype=tf.float64)
    input_matrix = tf.zeros((batch_size*sequence_size,128),dtype=tf.float64)
    # data_rel = data[0]
    # matrix = tf.reshape(data_rel, [data_rel.shape[0] * data_rel.shape[1], 128])
    # input_matrix = np.roll(matrix, 1, axis=0).reshape((data_rel.shape[0], data_rel.shape[1], 128))
    for index in range(0,100):
        z_output = tf.reshape(z_output, [batch_size * sequence_size, 32])
        input_matrix = tf.reshape(input_matrix, [batch_size * sequence_size, 128])
        decoder_input = tf.concat([input_matrix, z_output], 1)
        decoder_input = tf.reshape(decoder_input, (batch_size, sequence_size, 160))
        tf.print(decoder_input[0:5],summarize=100000)
        output, dec_states = tf.nn.dynamic_rnn(decoder_model,decoder_input,dtype=tf.float64)
        output = tf.nn.softmax(output)
        output = tf.reshape(output,[output.shape[0]*output.shape[1],128])
        #load_data(output.numpy())
        preds = output.numpy()
        #probas = np.random.multinomial(1,preds[index],1)
        #probas = np.random.choice(np.arange(0,128),p=output[index+1].numpy())
        probas = tf.argmax(output[index+1])
        input_matrix = tf.reshape(input_matrix, [batch_size * sequence_size, 128]).numpy()
        input_matrix[index+1] = tf.one_hot(probas,depth=128)
        #input_matrix[index+1] = probas
        input_matrix = tf.reshape(tf.convert_to_tensor(input_matrix), [batch_size,sequence_size,128])
        print("Generating data: ",index, " from 4000")
    print(input_matrix)
    input_matrix = tf.reshape(input_matrix,[batch_size * sequence_size, 128])
    load_data(input_matrix.numpy())
    gt = input_matrix[0]
    piano_roll_to_midi(tf.cast(input_matrix,tf.float32), 20, "bach_compare_goods_autoregress.mid")


def load_data(data):
    #c = np.loadtxt("test.txt")
    # piano_roll_to_midi(Y,20)
    newarray = []
    for index,val in enumerate(data):
        if max(val) > 0:
            newarray.append(tf.argmax(val))
        else:
            newarray.append(-1)
    newarray = tf.convert_to_tensor(newarray)
    Y = tf.one_hot(tf.convert_to_tensor(newarray),depth=128)
    #Y = tf.one_hot(tf.argmax(c, dimension=1), depth=128)
    plot_sequences(Y)
    #Y = Y[..., :-1]
    #Y = np.asarray([fgt[0:-1] for fgt in Y])
    #tgt=Y[10]
    piano_roll_to_midi(Y,20,"bach_compare_goodsall10.mid")


tf.enable_eager_execution()
#autoencoder()
#eager_autoencoder()
#load_data(0)
#eager_autoregressive("./log/model/autoregressive_sectest")
generate(64,64)