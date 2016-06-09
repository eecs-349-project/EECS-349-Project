import matplotlib.pyplot as plt

import numpy as np

import tempfile
import tensorflow as tf

from tf_rl.controller import DiscreteDeepQ
from tf_rl.simulation.quadrotor_simple import Quadrotor
from tf_rl.models import MLP

import scipy.io as sio

import copy

N = Quadrotor.num_of_actions

tf.reset_default_graph()
session = tf.InteractiveSession()

LOG_DIR = tempfile.mkdtemp()
print(LOG_DIR)
journalist = tf.train.SummaryWriter(LOG_DIR)

brain = MLP([4, ], [32, 64,  N], [tf.tanh, tf.tanh, tf.identity])

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)

current_controller = DiscreteDeepQ(4, N, brain, optimizer, session,
                                   discount_rate=0.9, exploration_period=100000,
                                   max_experience=10000,
                                   minibatch_size=64,
                                   random_action_probability=0.05,
                                   store_every_nth=1, train_every_nth=4,
                                   target_network_update_rate=0.1,
                                   summary_writer=journalist)

session.run(tf.initialize_all_variables())
session.run(current_controller.target_network_update)
journalist.add_graph(session.graph_def)

performances = []

dt = 0.02
try:
    for game_idx in range(10000):
        game = Quadrotor()
        game_iterations = 0

        observation = game.observe()
        x0 = copy.deepcopy(observation)
        rewards = []
        cost0 = game.cost()
        path = [copy.deepcopy(observation)]
        while game_iterations < 100 and not game.is_over():
            action = current_controller.action(observation)
            game.perform_action(action)
            game.step(dt)
            cost1 = game.cost()

            reward = cost0-cost1 - 2
            # reward = -reward
            rewards.append(reward)
            new_observation = game.observe()
            current_controller.store(observation, action, reward, new_observation)
            current_controller.training_step()

            observation = new_observation
            cost0 = cost1

            game_iterations += 1

            path.append(copy.deepcopy(observation))

        sio.savemat('/home/fantaosha/Documents/tensorflow-deepq/results/quadrotor_path/quadrotor_'+str(game_idx)+'.mat',
                    {'path': np.array(path)})
        performance = np.sum(rewards)
        performances.append(performance)

        print "\r================================================================================="
        print "\rGame %d: iterations before success %d." % (game_idx, game_iterations)
        print "Rewards: ",
        print np.sum(rewards)
        print "\r",
        print np.append(x0, observation)

        if game_idx % 100 == 0:
            saver = tf.train.Saver()
            save_path = saver.save(current_controller.s,
                                   "./saved_models/point/model_"+str(game_idx)+".ckpt")

        if game_idx % 500 == 0:
            N = 500
            smooth_performances = [float(sum(performances[i:i+N])) / N
                                   for i in range(0, len(performances) - N)]
            plt.clf()
            plt.plot(range(len(smooth_performances)), smooth_performances)
            plt.draw()
            plt.show(block=False)

except KeyboardInterrupt:
    print "Interrupted"

N = 500
smooth_performances = [float(sum(performances[i:i+N])) / N
                       for i in range(0, len(performances) - N)]

plt.plot(range(len(smooth_performances)), smooth_performances)
