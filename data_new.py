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

def eager_autoencoder():
    learning_rate = 0.001
    num_steps = 1000
    batch_size = 100
    display_step = 20
    examples_to_shop = 10

    num_hidden_1 = 128  # 1st layer num features
    num_hidden_2 = 16  # 2nd layer num features (the latent dim)
    num_input = 128  # MNIST data input (img shape: 28*28)

    # tf Graph input (only pictures)
    # X = tf.placeholder("float", [None, num_input])

    files = glob.glob(".\\midi\\bach\\test" + '/**/*.mid', recursive=True)
    print(files)
    test = eager(files, 64, 64)
    print("DATA PROCESSING COMPLETE")
    # Construct model
    model, init = model_layers(None, None,True)
    train_writer = tf.contrib.summary.create_file_writer('./log/good_smallbatch',flush_millis=1000)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    checkpoint_dir = './log/model/good_smallbatch/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    #root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    for epoch in range(num_steps):
        avg_loss = 0
        size = len(test)
        for index in range(0,size):
            with tf.GradientTape() as tape:
                #t = test[0][i]
                current_pred, new_states = tf.nn.dynamic_rnn(model, test[index], dtype=tf.float64)
                #
                # # Prediction
                y_pred = current_pred
                #y_pred = current_pred.numpy()
                # for i,batch in enumerate(y_pred):
                #     for j,seq in enumerate(batch):
                #         y_pred[i,j] = multiply(seq)

                #model, init = model_layers(None,None,False)
                #current_pred, new_states = tf.nn.dynamic_rnn(model, y_pred, dtype=tf.float64)
                #y_pred = tf.nn.softmax(y_pred)
                #matrix = tf.reshape(y_pred, [y_pred.shape[0] * y_pred.shape[1], 128])
                #np.savetxt("test.txt", matrix.numpy())
                # # Targets (Labels) are the input data.
                y_true = test[index]
                #
                # # Define loss and optimizer, minimize the squared error
                loss = tf.losses.softmax_cross_entropy(y_true, y_pred)
                grads = tape.gradient(loss,model.trainable_variables)
                optimizer.apply_gradients(zip(grads,model.trainable_variables))
                with train_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("loss", loss, step=index)
                    tf.contrib.summary.scalar("step", index, step=index)
                print(epoch,loss)
        if epoch % 20 == 0:
            checkpoint_dir = './log/model/good_smallbatch/'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            root.save(checkpoint_prefix)
            matrix = tf.reshape(y_pred, [current_pred.shape[0] * current_pred.shape[1], 128])
            t = matrix[0].numpy()
            # matrix = tf.argmax(current_pred,dimension=2)
            # matrix = tf.reshape(matrix,[-1])
            #np.savetxt("test_multfiles.txt",matrix)
    np.savetxt("test_smallbatch.txt", matrix)
    # matrix = tf.reshape(y_pred, [current_pred.shape[0] * current_pred.shape[1], 128])
    # t = matrix[0].numpy()
    # # matrix = tf.argmax(current_pred,dimension=2)
    # # matrix = tf.reshape(matrix,[-1])
    # Y = tf.one_hot(tf.argmax(matrix, dimension=1), depth=128)
    # yt = Y[0].numpy()
    # # piano_roll_to_midi(Y,20)
    # plot_sequences(Y)
    # #Y = Y[..., :-1]
    # #Y = np.asarray([fgt[0:-1] for fgt in Y])
    # piano_roll_to_midi(Y,20,"bach_compare_goodt.mid")


            #             print('Step %i: Minibatch Loss: %f' % (i, l))
    # init_op = iterator.initializer
    # init_val = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    # # Start Training
    # # Start a new TF session
    # with tf.Session() as sess:
    #
    #     # Run the initializer
    #     train_writer = tf.summary.FileWriter('./log',
    #                                          sess.graph)
    #     sess.run(init_op)
    #     sess.run(init_val)
    #     # Training
    #     states = np.zeros((3, 2, 64, 128))
    #     for i in range(1, num_steps + 1):
    #         # Run optimization op (backprop) and cost op (to get loss value)
    #         merge = tf.summary.merge_all()
    #         summary, _, l, f = sess.run([merge, optimizer, loss, p])
    #         # Display logs per step
    #         if i % display_step == 0 or i == 1:
    #             print('Step %i: Minibatch Loss: %f' % (i, l))
    #         train_writer.add_summary(summary, i)
    #     save_model(sess)

def autoencoder():
    learning_rate = 0.0001
    num_steps = 3000
    batch_size = 100
    display_step = 20
    examples_to_shop = 10

    num_hidden_1 = 128 # 1st layer num features
    num_hidden_2 = 16 # 2nd layer num features (the latent dim)
    num_input = 128 # MNIST data input (img shape: 28*28)

    # tf Graph input (only pictures)
    #X = tf.placeholder("float", [None, num_input])

    files = glob.glob(".\\midi" + '/**/*.mid', recursive=True)
    print(files)
    test = eager(files, 64, 64)
    iterator = test.make_initializable_iterator()
    roll = iterator.get_next()
    p = tf.print(roll[0][0])
    # Construct model
    init_state = tf.placeholder(tf.float64, [3, 2, 64, 128])
    model,init = model_layers(roll,init_state)
    current_pred, new_states = tf.nn.dynamic_rnn(model, roll, dtype=tf.float64,initial_state=init)
    #
    # # Prediction
    y_pred = current_pred
    # # Targets (Labels) are the input data.
    y_true = roll
    #
    # # Define loss and optimizer, minimize the squared error
    loss = tf.losses.softmax_cross_entropy(roll,y_pred)
    tf.summary.scalar("loss",loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = iterator.initializer
    init_val= tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Start Training
    # Start a new TF session
    with tf.Session() as sess:

        # Run the initializer
        train_writer = tf.summary.FileWriter('./log',
                                             sess.graph)
        sess.run(init_op)
        sess.run(init_val)
        # Training
        states = np.zeros((3, 2, 64, 128))
        for i in range(1, num_steps + 1):
            # Run optimization op (backprop) and cost op (to get loss value)
            merge = tf.summary.merge_all()
            summary,_,l,f= sess.run([merge,optimizer,loss,p])
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))
            train_writer.add_summary(summary,i)
        save_model(sess)

def model_layers(x,init,enc):
      # None is for batch_size
    #
    # state_per_layer_list = tf.unstack(init, axis=0)
    # #
    # rnn_tuple_state = tuple(
    #     [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],
    #                                    state_per_layer_list[idx][1])
    #      for idx in range(3)]
    # )
    # cell_state = tf.placeholder(tf.float32, [64, 128])
    # hidden_state = tf.placeholder(tf.float32, [64, 128])
    # init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
    #rnn_tuple_state = tf.nn.rnn_cell.LSTMCell(128).zero_state(64,dtype=tf.float64)
    #
    # #multi = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size) for size in [256, 256, 256]], state_is_tuple=True)
    #cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [128,256,256,16,256,256,128]])
    # if enc:
    #     cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [128,32]])
    # else:
    #     cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [32,128]])
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [128,32,128]])
    init = cell.zero_state(32, tf.float64)
    return cell,init

def lstm_cell(state_value):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_value, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    # if state_value == 32:
    #     t1 = np.linspace(1,0,16,endpoint=False)
    #     t2 = np.linspace(0,1,16)
    #     t3 = np.append(t1,t2)
    #     cell = tf.math.multiply(cell,t3)
    return cell

def multiply(t):
    t1 = np.linspace(1,0,16,endpoint=False)
    t2 = np.linspace(0,1,16)
    t3 = np.append(t1,t2)
    test = tf.math.multiply(t,t3)
    return test

def save_model(sess):
    tf.saved_model.simple_save(
        sess, 'log/model/test/model.ckpt',
    )

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
    # model, init = model_layers(None, None)
    # checkpoint_dir = './log/model/test/'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # root = tf.train.Checkpoint(model=model,
    #                            )
    # root.restore(tf.train.latest_checkpoint(checkpoint_prefix))
    # files = glob.glob(".\\midi\\bach\\test" + '/**/*.mid', recursive=True)
    # test = eager(files, 64, 64)
    # init = model.zero_state(64, tf.float64)
    # current_pred, new_states = tf.nn.dynamic_rnn(model, test[0], dtype=tf.float64)
    # matrix = tf.reshape(current_pred,[current_pred.shape[0]*current_pred.shape[1],129])
    # # matrix = tf.argmax(current_pred,dimension=2)
    # # matrix = tf.reshape(matrix,[-1])
    # Y = tf.one_hot(tf.argmax(matrix, dimension=1), depth=129)
    # #piano_roll_to_midi(Y,20)
    # plot_sequences(Y)
    # print(tf.losses.softmax_cross_entropy(test[0],current_pred))

tf.enable_eager_execution()
#autoencoder()
eager_autoencoder()
#load_model()