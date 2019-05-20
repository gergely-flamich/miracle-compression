import numpy as np

import tensorflow as tf
from sonnet import AbstractModule, Linear, BatchFlatten, BatchReshape, reuse_variables, \
    Conv2D, BatchNorm
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
