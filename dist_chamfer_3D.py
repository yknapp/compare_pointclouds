from torch import nn
from torch.autograd import Function
import torch
import importlib
import os
chamfer_found = importlib.find_loader("chamfer_3D") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load
    chamfer_3D = load(name="chamfer_3D",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
              ])
    print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D
    print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


# class chamfer_3DDist(nn.Module):
#     def __init__(self):
#         super(chamfer_3DDist, self).__init__()
#
#     def forward(self, input1, input2):
#         input1 = input1.contiguous()
#         input2 = input2.contiguous()
#         return chamfer_3DFunction.apply(input1, input2)

# Felix' implementation
# class chamfer_3DDist(nn.Module):
#     def __init__(self):
#         super(chamfer_3DDist, self).__init__()
#
#     def forward(self, input1, input2):
#         input1 = input1.contiguous()
#         input2 = input2.contiguous()
#         inp = torch.cat((input1, input2)).unsqueeze(1)
#         out = torch.zeros(inp.size(0),inp.size(0))
#         for i in range(inp.size(0)):
#             for j in range(inp.size(0)):
#                 if i < j:
#                     dist1, dist2, idx1, idx2 = chamfer_3DFunction.apply(inp[i], inp[j])
#                     out[i,j] = (torch.mean(dist1, 1)) + (torch.mean(dist2, 1))
#                 else:
#                     out[i,j] = out[j,i]
#         return out


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        inp = input1 + input2
        output = torch.zeros(len(inp), len(inp))

        # calculate chamfer distance between every pointcloud of domain 1 and domain 2 and save it to output matrix
        for i in range(len(inp)):
            for j in range(len(inp)):
                #print("i: %s, j: %s, total: %s" % (i, j, len(inp)))
                if i < j:
                    dist1, dist2, idx1, idx2 = chamfer_3DFunction.apply(inp[i].contiguous(), inp[j].contiguous())
                    output[i,j] = (torch.mean(dist1, 1)) + (torch.mean(dist2, 1))
                else:
                    output[i,j] = output[j,i]
        #print("OUTPUT: ", output)
        return output
