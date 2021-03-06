# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import warnings
import os
from distutils.util import strtobool

from horovod.common.util import check_extension

try:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib_v2')
except:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch import compression
from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import allgather, allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import join
from horovod.torch.mpi_ops import poll, synchronize
from horovod.torch.mpi_ops import init, shutdown
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.torch.mpi_ops import gloo_enabled, gloo_built
from horovod.torch.mpi_ops import nccl_built, ddl_built, mlsl_built

import torch
import collections


class Communicator(object):
    def async_send(self, tensors, name):
        pass

    def wait_receive(self, handles, ctx):
        pass


class Allreduce(Communicator):
    def __init__(self, compressor):
        self.compressor = compressor

    def async_send(self, tensors_compressed, name):
        handles = []
        for i, tensor_compressed in enumerate(tensors_compressed):
            handles.append(allreduce_async_(tensor_compressed, self.compressor.average, name + str(i)))
        return handles

    def wait_receive(self, handles, ctx):
        output = [synchronize(h) for h in handles]
        return self.compressor.decompress(output, ctx)


class Allgather(Communicator):
    def __init__(self, compressor, horovod_size):
        self.horovod_size = horovod_size
        self.compressor = compressor

    def async_send(self, tensors_compressed, name):
        """
        :param tensors_compressed: list of flat tensors to communicate
        :param name: for the all_gather operation
        :return: handles to synchronize, tensor sizes per rank
        """
        tensors_size = [t.numel() for t in tensors_compressed]  # list of tensor size for this rank
        if self.compressor.tensors_size_are_same:
            tensors_size_ag = [tensors_size] * self.horovod_size  # list of tensor sizes per rank
            tensor_sizes = zip(*tensors_size_ag)  # transpose
        else:
            tensors_size = torch.tensor(tensors_size)  # TODO: set device
            gathered = allgather(tensors_size)  # tensor of tensor sizes per rank
            tensor_sizes = gathered.view([self.horovod_size, -1]).t().tolist()  # transpose, to list

        handles = []
        for tensor_compressed in tensors_compressed:
            handle = allgather_async(tensor_compressed)
            handles.append(handle)

        return handles, tensor_sizes

    def wait_receive(self, result, ctx):
        handles, tensor_sizes = result
        tensors_ag = []
        for handle, sizes in zip(handles, tensor_sizes):
            gathered = synchronize(handle)
            tensors_ag.append(gathered.split(sizes))

        list_tensor_decompressed = []
        for tensor_compressed in zip(*tensors_ag):
            tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx)
            list_tensor_decompressed.append(tensor_decompressed)

        tensors_aggregated = self.compressor.aggregate(list_tensor_decompressed)
        return (tensors_aggregated / self.horovod_size) if self.compressor.average else tensors_aggregated


class Broadcast(Communicator):
    def __init__(self, compressor, horovod_size):
        self.horovod_size = horovod_size
        self.compressor = compressor

    def async_send(self, tensors_compressed, name):
        handles = []
        for root_rank in range(self.horovod_size):
            rank_handles = []
            for i, tensor_compressed in enumerate(tensors_compressed):
                rank_handles.append(broadcast_async(tensor_compressed, root_rank, name + str(root_rank) + '_' + str(i)))
            handles.append(rank_handles)
        return handles

    def wait_receive(self, handles, ctx):
        tensors_decompressed = []
        for ranki in handles:
            tensors_compressed = [synchronize(h) for h in ranki]
            tensor_decompressed = self.compressor.decompress(tensors_compressed, ctx)
            tensors_decompressed.append(tensor_decompressed)
        tensor_aggregated = self.compressor.aggregate(tensors_decompressed)
        return (tensor_aggregated / self.horovod_size) if self.compressor.average else tensor_aggregated


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compressor=None, communication='allreduce',
                 backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)

        communication = os.environ.get('HOROVOD_COMM_METHOD', 'allreduce')

        if compressor is None:
            params = {
                "compress_method": os.environ.get('HOROVOD_COMPRESS_METHOD', 'none'),
                "use_memory": strtobool(os.environ.get('HOROVOD_USE_MEMORY', "False")),
                "compress_ratio": float(os.environ.get('HOROVOD_COMPRESS_RATIO', 0.1)),
                "threshold_val": float(os.environ.get('HOROVOD_THRESHOLD_VAL', 0.01)),
                "quantum_num": int(os.environ.get('HOROVOD_QUANTUM_NUM', 256)),
                "gradient_clipping": strtobool(os.environ.get('HOROVOD_GRADIENT_CLIPPING', "False")),
                "momentum": float(os.environ.get('HOROVOD_MOMENTUM', 0.9)),
                "learning_rate": float(os.environ.get('HOROVOD_INIT_LR', 0.1)),
                "beta": float(os.environ.get('HOROVOD_MEMORY_BETA', 1.0)),
                "gamma": float(os.environ.get('HOROVOD_MEMORY_GAMMA', 1.0)),
                'compress_rank': int(os.environ.get('HOROVOD_COMPRESS_RANK', 2))
            }

            if params["compress_method"] in ["none", "efsignsgd", "dgc", "powersgd"]:
                pass
            elif params["use_memory"]:
                memory = compression.ResidualMemory(beta=params["beta"], gamma=params["gamma"])
            else:
                memory = compression.NoneMemory()

            if params["compress_method"] == 'none':
                self.compressor = compression.NoneCompressor()
            elif params["compress_method"] == 'fp16':
                self.compressor = compression.FP16Compressor(memory=memory)
            elif params["compress_method"] == 'randomk':
                self.compressor = compression.RandomKCompressor(compress_ratio=params['compress_ratio'],
                                                                memory=memory)
            elif params["compress_method"] == 'topk':
                self.compressor = compression.TopKCompressor(compress_ratio=params['compress_ratio'],
                                                             memory=memory)
            elif params["compress_method"] == 'threshold':
                self.compressor = compression.ThresholdCompressor(threshold_val=params['threshold_val'],
                                                                  memory=memory)
            elif params["compress_method"] == 'signsgd':
                self.compressor = compression.SignSGDCompressor(memory=memory)
            elif params["compress_method"] == 'efsignsgd':
                self.compressor = compression.EFSignSGDCompressor(lr=params['learning_rate'])
            elif params["compress_method"] == 'signum':
                self.compressor = compression.SignumCompressor(momentum=params['momentum'], memory=memory)
            elif params["compress_method"] == 'qsgd':
                self.compressor = compression.QSGDCompressor(quantum_num=params['quantum_num'], memory=memory)
            elif params["compress_method"] == 'onebit':
                self.compressor = compression.OneBitCompressor(memory=memory)
            elif params["compress_method"] == 'terngrad':
                self.compressor = compression.TernGradCompressor(memory=memory)
            elif params["compress_method"] == 'dgc':
                self.compressor = compression.DgcCompressor(compress_ratio=params['compress_ratio'], 
                                                            momentum=params['momentum'],
                                                            gradient_clipping=params['gradient_clipping'])
            elif params["compress_method"] == 'powersgd':
                self.compressor = compression.PowerSGDCompressor(rank=params['compress_rank'],
                                                                 use_memory=params['use_memory'])
        else:
            self.compressor = compressor

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [('allreduce.noname.%s' % i, v)
                                for param_group in self.param_groups
                                for i, v in enumerate(param_group['params'])]

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids = {id(v)
                         for param_group in self.param_groups
                         for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self.horovod_size = size()
        if self.horovod_size > 1:
            self._register_hooks()
            if communication == 'allreduce':
                self.communication = Allreduce(self.compressor)
            elif communication == 'allgather':
                self.communication = Allgather(self.compressor, self.horovod_size)
            elif communication == 'broadcast':
                self.communication = Broadcast(self.compressor, self.horovod_size)
            else:  # error
                self.communication = None

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _communicate_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad
        tensor = self.compressor.memory_compensate(tensor, name)
        tensors_compressed, ctx = self.compressor.compress(tensor, name)
        self.compressor.memory_update(tensor, name, tensors_compressed, ctx)

        handles = self.communication.async_send(tensors_compressed, name)
        return handles, ctx

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._communicate_grad_async(p)
            self._handles[p] = (handle, ctx)

        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            self._handles[p] = self._communicate_grad_async(p)

        for p, value in self._handles.items():
            handles, ctx = value
            if handles is None:
                handles, ctx = self._communicate_grad_async(p)
                self._handles[p] = (handles, ctx)
        for p, value in self._handles.items():
            handles, ctx = value
            tensor = self.communication.wait_receive(handles, ctx)
            self._allreduce_delay[p] = self.backward_passes_per_step
            p.grad.set_(tensor)
        self._handles.clear()

        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn("optimizer.step() called without "
                              "optimizer.skip_synchronize() context after "
                              "optimizer.synchronize(). This can cause training "
                              "slowdown. You may want to consider using "
                              "optimizer.skip_synchronize() context if you use "
                              "optimizer.synchronize() in your code.")
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return super(self.__class__, self).zero_grad()


def DistributedOptimizer(optimizer, named_parameters=None,
                         compressor=None,
                         communication='allreduce',
                         backward_passes_per_step=1):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by ``loss.backward()``
    in parallel with each other. The ``step()`` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the ``synchronize()`` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before ``step()`` is executed.
    Make sure to use ``optimizer.skip_synchronize()`` if you're calling ``synchronize()``
    in your code.

    Example of gradient clipping:

    .. code-block:: python

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        with optimizer.skip_synchronize():
            optimizer.step()

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compressor: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        communication: 'allgather', 'broadcast', 'allreduce'
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compressor, communication, backward_passes_per_step)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the ``model.state_dict()``,
    ``model.named_parameters()``, or ``model.parameters()``.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.cpu().numpy()[0])

        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.cpu().numpy()[0], dtypes)

        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
