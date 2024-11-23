import torch
import torch.nn as nn
from torch.autograd import Function
from utils.find_layers import replace_all_layer_type_recursive
import numpy as np
#
class GuidedBackpropReLU(Function):
    @staticmethod
    # ctx is a context object that can be used to stash information for backward computation
    # input is a tensor containing the input to the ReLU functions
    def forward(self, input_img):
        # get the positive mask
        # 1 will be the positive mask
        # 0 will be the negative mask
        positive_mask = (input_img > 0).type_as(input_img)

        # get the output of the forward pass for the ReLU
        #  element-wise multiplication
        # torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
        #  out = input + value * (tensor1 * tensor2)
        output = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            input_img,
            positive_mask,
        )

        # save the tensor for the backward pass
        self.save_for_backward(input_img, output)

        return output

    @staticmethod
    # ctx is a context object that can be used to stash information for backward computation
    # grad_output is the gradient of the loss with respect to the output of the ReLU
    def backward(self, grad_output):
        # get the input and output tensors from the forward pass
        input_img, output = self.saved_tensors
        # initialize the gradient of the input tensor to None
        grad_input = None

        # get the positive mask for the input tensor
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        # get the positive mask for the output tensor
        positive_mask_2 = (grad_output > 0).type_as(grad_output)

        # compute the gradient of the input tensor
        grad_input = torch.addcmul(
            torch.zeros(
                input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(
                    input_img.size()).type_as(input_img),
                grad_output,
                positive_mask_1,
            ),
            positive_mask_2,
        )

        return grad_input




class GuidedBackpropReLUasModule(torch.nn.Module):
    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU().apply(input_img)



class GuidedBackpropReLUModel:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        # self.model.parameters() returns an iterator over module parameters
        # next() get the first element of the iterator
        # get the device info of the model
        self.device = next(self.model.parameters()).device

    def forward(self, input_img):
        return self.model(input_img)


    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model, nn.ReLU, GuidedBackpropReLUasModule())


        input_img = input_img.to(self.device)
        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        loss = output[0, target_category]
        loss.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))

        replace_all_layer_type_recursive(self.model, GuidedBackpropReLUasModule, nn.ReLU())

        return output






