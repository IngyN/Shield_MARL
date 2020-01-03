import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

scenario = scenarios.load("simple" + ".py").Scenario()
world = scenario.make_world()
trainer = MADDPGAgentTrainer

# TODO actually train.