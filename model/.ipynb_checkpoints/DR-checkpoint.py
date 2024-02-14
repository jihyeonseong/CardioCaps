import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, args, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3, affine=None, coupling=None):
        super(DenseCapsule, self).__init__()
        self.args = args
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.affine = affine
        self.coupling = coupling
        
        if args.affine == 'param':
            self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        elif args.affine == 'shared':
            self.weight = nn.Conv2d(1, 2, 1, 1)
        elif args.affine == 'constant':
            self.weight = 0.01 * torch.ones(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps).cuda()
        else:
            raise ValueError(f"Passed undefined affine : {cfg.affine}")
            
        #self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        self.attn = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        """
        
        if self.args.affine == 'shared':
            x_hat = self.weight(x.unsqueeze(1)).repeat(1,1,1,2)
        else:
            x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
            
        #x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        """
        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        """
        x_hat_detached = x_hat.detach()
        
        """
        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        """
        if self.args.dr == True:
            b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).cuda())

            assert self.routings > 0, 'The \'routings\' should be > 0.'
            for i in range(self.routings):
                # c.size = [batch, out_num_caps, in_num_caps]
                c = F.softmax(b, dim=1)

                # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
                if i == self.routings - 1:
                    """
                    # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                    # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                    # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                    """
                    outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                    outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                    """
                    # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                    # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                    # => b.size          =[batch, out_num_caps, in_num_caps]
                    """
                    b = b + torch.sum(outputs * x_hat_detached, dim=-1)
        else:
            c = F.softmax(self.attn(x_hat), dim=1)
            outputs = squash(torch.sum(c * x_hat, dim=-2, keepdim=True))

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)
    
    
class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, args, input_size, classes, routings, affine=None, coupling=None):
        super(CapsuleNet, self).__init__()
        self.args = args
        self.input_size = input_size
        self.classes = classes
        self.routings = routings
        self.affine = affine
        self.coupling = coupling

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 8*32, kernel_size=(9,9), stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(8*32, 8*32, 8, kernel_size=(9,9), stride=(2,2), padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(args = args, in_num_caps=32*14400, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings,
                                      affine=self.affine, coupling=self.coupling) # replacing affine matrix with 'uniform' or ;random' matrix
                                     # replacing coupling coefficient with 'uniform' or 'random' matrix

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * input_size[1] * input_size[2]),
            #nn.Sigmoid()
        )
        
        self.regression = nn.Linear(3 * input_size[1] * input_size[2], 1)

        self.relu = nn.ReLU()

    def forward(self, x, y=None): 
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        width = self.regression(reconstruction)
        
        return length, reconstruction.view(-1, 3*self.input_size[1]*self.input_size[2]), width