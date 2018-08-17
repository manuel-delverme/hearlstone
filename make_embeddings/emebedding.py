#!/usr/bin/env python3.5
import shutil
import torch.optim

import os
import torch
import torch.utils.data
import torch.autograd
import torch
import glob
import tqdm
import bz2
import simple_env
import tensorflow as tf
import pickle
import fireplace
import random
import numpy as np
import fireplace.logging
from fireplace.exceptions import GameOver
import torch.utils.data
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    return parser.parse_args()


class HSgames_dataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        # self.files = sorted(glob.glob(data_file), reverse=True)
        self.states = []
        while True:
            with bz2.BZ2File(data_file, "rb") as fin:
                try:
                    training_tuples = pickle.load(fin)
                except EOFError:
                    break
                for training_tuple in training_tuples:
                    s, a, s1 = training_tuple
                    self.states.append(s1)

    def __getitem__(self, index):
        game_state = self.states[index]
        return game_state

    def __len__(self):
        return len(self.states)


def load_dataset(data_file, batch_size, preprocess=None):
    dataset = HSgames_dataset(data_file=data_file)
    if preprocess:
        dataset = preprocess(dataset)
    states_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    return states_loader


def trainingset_length(data_file):
    length = 0
    while True:
        with bz2.BZ2File(data_file, "rb") as fin:
            try:
                training_tuples = pickle.load(fin)
            except EOFError:
                break
            length += len(training_tuples)
            print(length)
    return length


def load_batch(data_file, batch_size, max_games=10000):
    batch = []
    counter = 0
    while True:
        with bz2.BZ2File(data_file, "rb") as fin:
            try:
                training_tuples = pickle.load(fin)
            except EOFError:
                break
            for training_tuple in training_tuples:
                s, a, s1 = training_tuple
                batch.append(s1)
                if len(batch) == batch_size:
                    yield np.array(batch)
                    counter += batch_size
                    batch = []
                    if counter > max_games:
                        break
            if counter > max_games:
                break


def gather_transitions(num_transitions):
    file_idx = 0
    data_dump = "data/training_data{}.pbz"
    while True:
        try:
            with open(data_dump.format(file_idx), "rb") as _:
                pass
        except FileNotFoundError:
            break
        else:
            file_idx += 1
    generate_transitions(data_dump.format(file_idx), num_transitions)
    return data_dump.format(file_idx)


def generate_transitions(dump_file, nr_transitions):
    games_finished = 0
    nr_tuples = 0

    with bz2.BZ2File(dump_file, "wb") as fout:
        env = simple_env.HSEnv()
        progress = tqdm.tqdm(total=nr_transitions)
        while nr_tuples < nr_transitions:
            training_set = []
            for _ in range(10):
                old_s, reward, terminal, info = env.reset()
                done = False
                nr_new_samples = 0
                while not done:
                    possible_actions = info['possible_actions']
                    random_act = random.choice(possible_actions)
                    # print("action", random_act)
                    # try:
                    s, r, done, info = env.step(random_act)
                    # except Exception as e:
                    #    s, r, done, info = env.step(random_act)
                    training_tuple = (old_s, random_act.encode(), s)
                    # training_tuple = (old_s, None, s)
                    training_set.append(training_tuple)
                    old_s = s
                    nr_new_samples += 1
                games_finished += 1
                nr_tuples += nr_new_samples
                progress.update(nr_new_samples)
                progress.set_description("games_finished {}".format(games_finished))
            pickle.dump(training_set, fout)
            fout.flush()


def batch_data(files, batch_size):
    batch = []
    for data_file in files:  # range(100):
        with bz2.BZ2File(data_file, "rb") as fin:
            while True:
                try:
                    training_tuples = pickle.load(fin)
                except EOFError:
                    break

                for training_tuple in training_tuples:
                    s, a, s1 = training_tuple
                    batch.append(s1)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(12728, 4096),
            torch.nn.ReLU(True),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 16),
        )
        self.decoder = torch.nn.Sequential(*reversed([
            torch.nn.Linear(4096, 12728),
            torch.nn.ReLU(True),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(16, 32),
        ]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_state_embedding():
    args = parser_args()
    nr_epochs = 5
    # env = envs.simple_HSenv.simple_HSEnv(skip_mulligan=True)
    # model = AE(obs_dim=12728, config=cfg)
    model = Autoencoder()
    # nr_games = trainingset_length(
    #     data_file="data/training_data53.pbz",
    # )
    nr_games = 10240
    pb = tqdm.tqdm(total=nr_games * nr_epochs)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()  # cuda()

    if args.resume:
        checkpoint_file = "checkpoint.pth.tar"
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})" .format(checkpoint_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))

    model.train()
    for epoch in range(nr_epochs):
        print("epoch:", epoch)
        batch_loader = load_batch(
            data_file="data/training_data53.pbz",
            batch_size=256,
            max_games=102400,
        )
        for game_states in batch_loader:
            pb.update(len(game_states))
            game_states = torch.autograd.Variable(torch.FloatTensor(game_states))
            decoded = model(game_states)
            encoding_loss = loss_fn(decoded, game_states)
            pb.set_description("loss: {}".format(encoding_loss.data[0]))
            optimizer.zero_grad()
            encoding_loss.backward()
            optimizer.step()

        torch.save({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, 'checkpoint.pth.tar')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == "__main__":
    # fout = gather_transitions(10000)
    train_state_embedding()
