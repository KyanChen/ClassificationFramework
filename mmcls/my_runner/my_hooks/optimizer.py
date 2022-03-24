# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad

from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from mmcv.runner import allreduce_grads
from mmcv.runner import LossScaler, wrap_fp16_model
from mmcv.runner import HOOKS, Hook, OptimizerHook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class MyOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, detect_anomalous_params=False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner, pattern='siren'):
        runner.optimizer[pattern].zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer[pattern].step()

    def after_train_inner_iter(self, runner, pattern, loss):
        runner.optimizer[pattern].zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        loss.backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
        runner.optimizer[pattern].step()

    def detect_anomalous_parameters(self, loss, runner):
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')
