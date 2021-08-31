# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# code from: https://github.com/BorealisAI/advertorch
# see copyright in the Github folder

import os
import sys
from pathlib import Path

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import torch

from advertorch.bpda import BPDAWrapper
from advertorch_examples.utils import ROOT_PATH, mkdir

MODEL_PATH = os.path.join(ROOT_PATH, "madry_et_al_models")
mkdir(MODEL_PATH)
Path(os.path.join(MODEL_PATH, "__init__.py")).touch()
sys.path.append(MODEL_PATH)


class WrappedTfModel(object):
    def __init__(self, weights_path, model_class, scope="test"):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # New graph for each time this class is called, otherwise you get
        # variable collisions
        g = tf.Graph()
        with g.as_default():
            model = model_class()  # load model ops/vars in new graph
            saver = tf.train.Saver()  # grab vars from default graph
        sess = tf.Session(graph=g, config=config).__enter__()
        checkpoint = tf.train.latest_checkpoint(weights_path)
        saver.restore(sess, checkpoint)

        self.inputs = model.x_input
        self.logits = model.pre_softmax

        # self.session = tf.get_default_session()
        self.session = sess
        assert self.session.graph == self.inputs.graph

        with self.session.graph.as_default():
            self.bw_gradient_pre = tf.placeholder(tf.float32, self.logits.shape)
            bw_loss = tf.reduce_sum(self.logits * self.bw_gradient_pre)
            self.bw_gradients = tf.gradients(bw_loss, self.inputs)[0]

    def backward(self, inputs_val, logits_grad_val):
        inputs_grad_val = self.session.run(
            self.bw_gradients,
            feed_dict={self.inputs: inputs_val, self.bw_gradient_pre: logits_grad_val},
        )
        return inputs_grad_val

    def forward(self, inputs_val):
        logits_val = self.session.run(self.logits, feed_dict={self.inputs: inputs_val})
        return logits_val


class TorchWrappedModel(object):
    def __init__(self, tfmodel, device):
        self.tfmodel = tfmodel
        self.device = device

    def _to_numpy(self, val):
        return val.cpu().detach().numpy()

    def _to_torch(self, val):
        return torch.from_numpy(val).float().to(self.device)

    def forward(self, inputs_val):
        rval = self.tfmodel.forward(self._to_numpy(inputs_val))
        return self._to_torch(rval)

    def backward(self, inputs_val, logits_grad_val):
        rval = self.tfmodel.backward(
            self._to_numpy(inputs_val), self._to_numpy(logits_grad_val)
        )
        return self._to_torch(rval)


def load_madry_model(dataname, weights_path, device="cuda"):
    if dataname == "mnist":
        try:
            from .madry_mnist.model import Model

            print("madry_mnist found and imported")
        except (ImportError, ModuleNotFoundError):
            print(
                "madry_mnist not found, please install madry challenge with install_madry_challenge.sh"
            )

        def _process_inputs_val(val):
            return val.view(val.shape[0], 784)

        def _process_grads_val(val):
            return val.view(val.shape[0], 1, 28, 28)

    elif dataname == "cifar":

        try:
            from .madry_cifar.model import Model

            print("madry_cifar found and imported")
        except (ImportError, ModuleNotFoundError):
            print(
                "madry_cifar not found, please install madry challenge with install_madry_challenge.sh"
            )

        from functools import partial

        Model = partial(Model, mode="eval")

        def _process_inputs_val(val):
            return 255.0 * val.permute(0, 2, 3, 1)

        def _process_grads_val(val):
            return val.permute(0, 3, 1, 2) / 255.0

    else:
        raise ValueError(dataname)

    def _wrap_forward(forward):
        def new_forward(inputs_val):
            return forward(_process_inputs_val(inputs_val))

        return new_forward

    def _wrap_backward(backward):
        def new_backward(inputs_val, logits_grad_val):
            return _process_grads_val(
                backward(_process_inputs_val(*inputs_val), *logits_grad_val)
            )

        return new_backward

    ptmodel = TorchWrappedModel(WrappedTfModel(weights_path, Model), device)
    model = BPDAWrapper(
        forward=_wrap_forward(ptmodel.forward),
        backward=_wrap_backward(ptmodel.backward),
    )

    return model
