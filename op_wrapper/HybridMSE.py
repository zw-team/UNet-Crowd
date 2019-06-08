import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Function
from torch.nn import Module


class HybridMSEFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 3:
            print("wrong input parameters number, check the input")
            return
        pred = args[0]
        gt = args[1]
        ctx.resolutions = args[2]
        loss = torch.pow(pred.sum() - gt.sum(), 2) / 160000
        ctx.save_for_backward(pred, gt)
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        pred, gt = ctx.saved_tensors
        grad_weights = pred - gt
        grad_pred = grad_weights
        for i in ctx.resolutions:
            ds = functional.avg_pool2d(grad_weights, kernel_size=i, stride=i)
            up = functional.interpolate(ds, scale_factor=i, mode='nearest')
            grad_pred += up
            
        grad_pred *= (torch.ones(pred.shape, dtype=torch.float32).cuda() *
                     grad_outputs[0])
        return grad_pred, None, None


class HybridMSELoss(Module):
    def __init__(self, resolutions):
        super(HybridMSELoss, self).__init__()
        self.resolutions = resolutions

    def forward(self, pred, gt):
        return HybridMSEFunction.apply(pred, gt, self.resolutions)