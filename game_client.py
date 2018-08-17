import envs.simple_HSenv
import pickle
import socket
import gym


class RemoteHSEnv(gym.Env):
    def __init__(self, skip_mulligan=False):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('', 31337))

    def reset(self):
        self.socket.sendall(b'r')
        state_0 = self.socket.recv(1024)

    def step(self, action):
        self.socket.sendall(b's')
        self.socket.sendall(pickle.dumps({'foo': 'bar'}))
        packet = b''
        msg = None
        while msg != '\r':
            msg = self.socket.recv(1)
            packet += msg

        state, reward, terminal, info = pickle.loads(packet)
        return state, reward, terminal, info

