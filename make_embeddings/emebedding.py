#!/usr/bin/env python3.5
import bz2
import hearthEnv
import tensorflow as tf
import pickle
import fireplace
import random

import numpy as np

import fireplace.logging
from fireplace.exceptions import GameOver


def main():
    games_finished = 0
    nr_tuples = 0
    file_idx = 0
    data_dump = "make_embeddings/data/training_data{}.pbz"
    while True:
        try:
            with open(data_dump.format(file_idx), "rb") as _:
                pass
        except FileNotFoundError:
            break
        else:
            file_idx += 1

    training_set = []
    with bz2.BZ2File(data_dump.format(file_idx), "wb") as fout:
        while True:
            env = hearthEnv.HSEnv()
            old_s, reward, terminal, info = env.reset()
            done = False
            step = 0
            while not done:
                step += 1
                possible_actions = info['possible_actions']
                random_act = random.choice(possible_actions)
                print("action", random_act)
                try:
                    s, r, done, info = env.step(random_act)
                except Exception as e:
                    s, r, done, info = env.step(random_act)
                training_tuple = (old_s, random_act, s)
                # training_tuple = (old_s, None, s)
                training_set.append(training_tuple)
                old_s = s
            games_finished += 1
            nr_tuples += len(training_set)
            print("games_finished", games_finished, "nr_tuples", nr_tuples)
            try:
                pickle.dump(training_set, fout)
            except Exception as e:
                pickle.dump(training_set, fout)
                print(e)
            fout.flush()
            training_set = []


"""
def __train():
    # with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    fireplace.cards.db.initialize()
    training_set = []

    # encoder_op = encoder(X)
    # decoder_op = decoder(encoder_op)

    y_pred = decoder_op
    y_true = X

    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOtimizer(0.001).minimize(loss)
    for i in range(1000):
        s0, a0 = training_set

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: s0})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

            # g = sess.run(decoder_op, feed_dict={X: batch_x})

        # if games_finished > 10:
        #     break
        nr_tuples += len(training_set)
        print(games_finished, nr_tuples)
        pickle.dump(training_set, fout)
        fout.flush()
        training_set = []
    except TypeError as e:
        print("game failed")
    except Exception as e:
        print(str(e))
        break
"""

if __name__ == "__main__":
    main()
