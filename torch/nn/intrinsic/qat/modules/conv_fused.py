from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.intrinsic
import torch.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

class _ConvBnNd(nn.modules.conv._ConvNd):
    def __init__(self,
                 # BN args
                 bn,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.bn = bn
        self.freeze_bn = freeze_bn if self.training else True
        self.num_features = out_channels
        self.activation_post_process = self.qconfig.activation()
        self.weight_fake_quant = self.qconfig.weight()
        self.reset_bn_parameters()
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def reset_running_stats(self):
        # TODO: handle this
        # self.running_mean.zero_()
        # self.running_var.fill_(1)
        # self.num_batches_tracked.zero_()
        pass

    def reset_bn_parameters(self):
        # TODO: handle this
        # self.reset_running_stats()
        # init.uniform_(self.gamma)
        # init.zeros_(self.beta)
        # TODO: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ConvBnNd, self).reset_parameters()
        # A hack to avoid resetting on undefined parameters
        # if hasattr(self, 'gamma'):
            # self.reset_bn_parameters()

    def update_bn_stats(self):
        # self.freeze_bn = False
        # TODO: verify that this doesn't break anything
        self.bn.train()
        return self

    def freeze_bn_stats(self):
        # self.freeze_bn = True
        self.bn.eval()
        return self

    def _forward(self, input):

        # TODO: remove the old version before landing (keeping for easier debugging)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        # if self.momentum is None:
            # exponential_average_factor = 0.0
        # else:
            # exponential_average_factor = self.momentum

        # if self.training and not self.freeze_bn and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            # if self.num_batches_tracked is not None:
                # self.num_batches_tracked += 1
                # if self.momentum is None:  # use cumulative moving average
                    # exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                # else:  # use exponential moving average
                    # exponential_average_factor = self.momentum

        # we use running statistics from the previous batch, so this is an
        # approximation of the approach mentioned in the whitepaper, but we only
        # need to do one convolution in this case instead of two
        # running_std = torch.sqrt(self.running_var + self.eps)
        # scale_factor = self.gamma / running_std

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        # this does not include the conv bias
        conv = self._conv_forward(input, self.weight_fake_quant(scaled_weight))
        conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape([1, -1, 1, 1])
        conv = self.bn(conv_orig)
        return conv

        # if self.training and not self.freeze_bn:
            # recovering original conv to get original batch_mean and batch_var
            # if self.bias is not None:
                # conv_orig = conv / scale_factor.reshape([1, -1, 1, 1]) + self.bias.reshape([1, -1, 1, 1])
            # else:
                # conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])

            # batch_mean = torch.mean(conv_orig, dim=[0, 2, 3])
            # batch_var = torch.var(conv_orig, dim=[0, 2, 3], unbiased=False)
            # n = float(conv_orig.numel() / conv_orig.size()[1])
            # unbiased_batch_var = batch_var * (n / (n - 1))
            # batch_rstd = torch.ones_like(batch_var, memory_format=torch.contiguous_format) / torch.sqrt(batch_var + self.eps)

            # conv = (self.gamma * batch_rstd).reshape([1, -1, 1, 1]) * conv_orig + \
                # (self.beta - self.gamma * batch_rstd * batch_mean).reshape([1, -1, 1, 1])
            # self.running_mean = exponential_average_factor * batch_mean.detach() + \
                # (1 - exponential_average_factor) * self.running_mean
            # self.running_var = exponential_average_factor * unbiased_batch_var.detach() + \
                # (1 - exponential_average_factor) * self.running_var
        # else:
            # if self.bias is None:
                # conv = conv + (self.beta - self.gamma * self.running_mean /
                               # running_std).reshape([1, -1, 1, 1])
            # else:
                # conv = conv + (self.gamma * (self.bias - self.running_mean) / running_std + self.beta).reshape([1, -1, 1, 1])
        # return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ConvBnNd, self).extra_repr()

    def forward(self, input):
        return self.activation_post_process(self._forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(bn, conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         False,
                         qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        return qat_convbn

class ConvBn2d(_ConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        activation_post_process: fake quant module for output activation
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvBn2d

    def __init__(self,
                 # BN args
                 bn,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(self, bn, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _pair(0), groups, bias, padding_mode,
                           freeze_bn, qconfig)

class ConvBnReLU2d(ConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvBnReLU2d

    def __init__(self,
                 # BN args
                 bn,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super(ConvBnReLU2d, self).__init__(bn, in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias,
                                           padding_mode,
                                           freeze_bn,
                                           qconfig)

    def forward(self, input):
        return self.activation_post_process(F.relu(ConvBn2d._forward(self, input)))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        return super(ConvBnReLU2d, cls).from_float(mod, qconfig)

class ConvReLU2d(nnqat.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for both output activation and weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = torch.nn.intrinsic.ConvReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode,
                                         qconfig=qconfig)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = self.qconfig.activation()
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        return self.activation_post_process(F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight))))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        return super(ConvReLU2d, cls).from_float(mod, qconfig)

def update_bn_stats(mod):
    if type(mod) in set([ConvBnReLU2d, ConvBn2d]):
        mod.update_bn_stats()

def freeze_bn_stats(mod):
    if type(mod) in set([ConvBnReLU2d, ConvBn2d]):
        mod.freeze_bn_stats()
