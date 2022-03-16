import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Floor(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the floor function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the round function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Clamp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the clamp function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None

class TorchBinarize(nn.Module):
    """ Binarizes a value in the range [-1,+1] to {-1,+1} """
    def __init__(self):
        super(TorchBinarize, self).__init__()

    def forward(self, input):
        """  clip to [-1,1] """
        input = Clamp.apply(input, -1.0, 1.0)
        """ rescale to [0,1] """
        input = (input+1.0) / 2.0
        """ round to {0,1} """
        input = Round.apply(input)
        """ rescale back to {-1,1} """
        input = input*2.0 - 1.0
        return input

class TorchRoundToBits(nn.Module):
    """ Quantize a tensor to a bitwidth larger than 1 """
    def __init__(self, bits=2):
        super(TorchRoundToBits, self).__init__()
        assert bits > 1, "RoundToBits is only used with bitwidth larger than 1."
        self.bits = bits
        self.epsilon = 1e-7

    def forward(self, input):
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply( input/scaling ,0.0, 1.0 )
        """ round the mantessa bits to the required precision """
        input = Round.apply(input * (2.0**self.bits-1.0)) / (2.0**self.bits-1.0)
        return input * sign, scaling

class TorchQuantize(nn.Module):
    """ 
    Quantize an input tensor to the fixed-point representation. 
        Args:
        input: Input tensor
        bits:  Number of bits in the fixed-point
    """
    def __init__(self, bits=0):
        super(TorchQuantize, self).__init__()
        if bits == 0:
            self.quantize = nn.Identity()
        elif bits == 1:
            self.quantize = TorchBinarize()
        else:
            self.quantize = TorchRoundToBits(bits)

    def forward(self, input):
        return self.quantize(input)
        
class QuantizedConv2d(nn.Conv2d): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', wbits=0, abits=0):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, 
                                              kernel_size, stride, 
                                              padding, dilation, groups, 
                                              bias, padding_mode)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.weight_rescale = \
            np.sqrt(1.0/(kernel_size**2 * in_channels)) if (wbits == 1) else 1.0

    def forward(self, input): 
        ######################## Original Conv2d ########################
        return F.conv2d(self.quantize_a(input),
                        self.quantize_w(self.weight) * self.weight_rescale,
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
         

        ######################## test crossbar architecture #############
         
        OU_size = 9*8
        num_crossbar = int(self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3] / OU_size)
         
        crossbar_inp_unf = torch.nn.functional.unfold(self.quantize_a(input), kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).reshape(input.shape[0], OU_size, num_crossbar, -1).permute(0,3,2,1) # (batch_size, feature_map_size, num_crossbar, crossbar_size) 
        #print('crossbar_inp_unf: {}'.format(crossbar_inp_unf.shape))
        
        crossbar_column_weight = self.quantize_w(self.weight).reshape(self.weight.shape[0], OU_size, num_crossbar).permute(1,2,0) # (crossbar_size, num_crossbar, Cout)
        #print('crossbar_column_weight: {}'.format(crossbar_column_weight.shape))
         
        test_out_unf = torch.einsum('ijlk,kln->injl', [crossbar_inp_unf, crossbar_column_weight]) # (batch_size, Cout, feature_map_size, num_crossbar)
        #print('test_out_unf: {}'.format(test_out_unf.shape)) 

        new_out_unf = torch.einsum('injl->inj', [test_out_unf]) #+ torch.normal(torch.tensor(10.0)) # (batch_size, Cout, feature_map_size)
        #print('test_out_unf: {}'.format(test_out_unf.shape))   
         
        new_out_unf = new_out_unf.view(new_out_unf.shape[0], self.out_channels, int(new_out_unf.shape[2] ** 0.5), int(new_out_unf.shape[2] ** 0.5))
         
        ########################################################################################################################################################
 
        print((torch.nn.functional.conv2d(self.quantize_a(input), self.quantize_w(self.weight), padding = self.padding, stride = self.stride, bias = self.bias) - new_out_unf).abs().max()) 
        
        return new_out_unf