import torch
import torch.nn as nn
import torch.nn.functional as F

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1



class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        x += res
        return self.relu(x)



class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)

class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3,3), stride=(1,2,2), padding=1),
                 ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)


class UNet_3D_3D(nn.Module):
    def __init__(self, n_inputs=4, joinType="concat", ks=5, dilation=1, voxelGridSize = 128, nf_out = 64):
        super().__init__()

        nf = [512, 256, 128, 64]
        ws = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        nh = [2, 4, 8, 16]
        self.joinType = joinType
        self.n_inputs = n_inputs

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.temporalAttn  = torch.nn.MultiheadAttention(voxelGridSize*2, 32)

        def SmoothNet(inc, ouc):
            return torch.nn.Sequential(
                Conv_3d(inc, ouc, kernel_size=3, stride=1, padding=1, batchnorm=False),
                ResBlock(ouc, kernel_size=3),
            )

        self.smooth_ll = SmoothNet(int(voxelGridSize/2), nf_out)
        self.smooth_l = SmoothNet(int(voxelGridSize/4), nf_out)
        self.smooth = SmoothNet(int(voxelGridSize/8), nf_out)

        self.maxPool_vv = torch.nn.MaxPool3d((1,1,4), stride=(1,1,4))
        self.maxPool_v = torch.nn.MaxPool3d((1,1,2), stride=(1,1,2))

        self.maxPool_lll = torch.nn.MaxPool3d((1,8,8), stride=(1,8,8))
        self.maxPool_ll = torch.nn.MaxPool3d((1,4,4), stride=(1,4,4))
        self.maxPool_l = torch.nn.MaxPool3d((1,2,2), stride=(1,2,2))

        self.predict_ll = SynBlock(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=True)
        self.predict_l = SynBlock(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=False)
        self.predict = SynBlock(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=False)

    def forward(self, frames, voxel):
        images = torch.stack(frames, dim=2)
        voxel = torch.stack(voxel, dim=2)
        Batch, Channel, Temporal, H, W = voxel.shape

        voxel = torch.reshape(voxel, (voxel.size(0), voxel.size(3)*voxel.size(4), voxel.size(1)*voxel.size(2)))
        x_0, _ = self.temporalAttn(voxel, voxel, voxel)
        x_0 = torch.reshape(voxel, (Batch, Channel, Temporal, H, W))

        dx_1_max = self.maxPool_vv(x_0.permute(0, 2, 3, 4, 1))
        dx_1_abs = self.maxPool_vv(torch.abs(x_0.permute(0, 2, 3, 4, 1)))

        diff_matrix_dx1 = torch.eq(dx_1_abs,dx_1_max)
        dx_1 = dx_1_max*diff_matrix_dx1 + -1*dx_1_abs*torch.logical_not(diff_matrix_dx1)


        dx_2_max = self.maxPool_v(x_0.permute(0, 2, 3, 4, 1))
        dx_2_abs = self.maxPool_v(torch.abs(x_0.permute(0, 2, 3, 4, 1)))
        diff_matrix_dx2 = torch.eq(dx_2_abs,dx_2_max)
        dx_2 = dx_2_max*diff_matrix_dx2 + -1*dx_2_abs*torch.logical_not(diff_matrix_dx2)
        
        dx_1 = dx_1.permute(0, 4, 1, 2, 3)
        dx_2 = dx_2.permute(0, 4, 1, 2, 3)
        dx_3 = x_0

        fea3 = self.smooth_ll(dx_3)
        fea2 = self.smooth_l(dx_2)
        fea1 = self.smooth(dx_1) 
        
        fea3_max = self.maxPool_lll(fea3)
        fea3_abs = self.maxPool_lll(torch.abs(fea3))
        diff_matrix_fea3 = torch.eq(fea3_max,fea3_abs)
        fea3 = fea3_max*diff_matrix_fea3 + -1*fea3_abs*torch.logical_not(diff_matrix_fea3)

        fea2_max = self.maxPool_ll(fea2)
        fea2_abs = self.maxPool_ll(torch.abs(fea2))
        diff_matrix_fea2 = torch.eq(fea2_max,fea2_abs)
        fea2 = fea2_max*diff_matrix_fea2 + -1*fea2_abs*torch.logical_not(diff_matrix_fea2)

        fea1_max = self.maxPool_l(fea1)
        fea1_abs = self.maxPool_l(torch.abs(fea1))
        diff_matrix_fea1 = torch.eq(fea1_max,fea1_abs)
        fea1 = fea1_max*diff_matrix_fea1 + -1*fea1_abs*torch.logical_not(diff_matrix_fea1)

        out_ll = self.predict_ll(fea3, frames, fea2.size()[-2:]) 

        out_l = self.predict_l(fea2, frames, fea1.size()[-2:])
        out_l = F.interpolate(out_ll, size=out_l.size()[-2:], mode='bilinear') + out_l

        out = self.predict(fea1, frames, (H,W))
        out = F.interpolate(out_l, size=out.size()[-2:], mode='bilinear') + out

        if self.training:
            return out_ll, out_l, out
        else:
            return out
            # return out_ll, out_l, out

class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input


class SynBlock(nn.Module):
    def __init__(self, n_inputs, nf, ks, dilation, norm_weight=True):
        super(SynBlock, self).__init__()

        def Subnet_offset(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                nn.Softmax(1) if norm_weight else nn.Identity()
            )

        def Subnet_occlusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=nf, out_channels=n_inputs, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.n_inputs = n_inputs
        self.kernel_size = ks
        self.kernel_pad = int(((ks - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        import common.cupy_module.adacof as adacof #TODO
        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        self.ModuleWeight = Subnet_weight(ks ** 2)
        self.ModuleAlpha = Subnet_offset(ks ** 2)
        self.ModuleBeta = Subnet_offset(ks ** 2)
        self.moduleOcclusion = Subnet_occlusion()

        self.feature_fuse = Conv_2d(nf * n_inputs, nf, kernel_size=1, stride=1, batchnorm=False, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, fea, frames, output_size):
        H, W = output_size

        occ = torch.cat(torch.unbind(fea, 1), 1)
        occ = self.lrelu(self.feature_fuse(occ))
        Occlusion = self.moduleOcclusion(occ, (H, W))

        B, C, T, cur_H, cur_W = fea.shape
        fea = fea.transpose(1, 2).reshape(B*T, C, cur_H, cur_W)
        weights = self.ModuleWeight(fea, (H, W)).view(B, T, -1, H, W)
        alphas = self.ModuleAlpha(fea, (H, W)).view(B, T, -1, H, W)
        betas = self.ModuleBeta(fea, (H, W)).view(B, T, -1, H, W)

        warp = []
        for i in range(self.n_inputs):
            weight = weights[:, i].contiguous()
            alpha = alphas[:, i].contiguous()
            beta = betas[:, i].contiguous()
            occ = Occlusion[:, i:i+1]
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode='bilinear')

            warp.append(
                occ * self.moduleAdaCoF(self.modulePad(frame), weight, alpha, beta, self.dilation)
            )

        framet = sum(warp)
        return framet

if __name__ == '__main__':
    model = UNet_3D_3D('unet_18', n_inputs=4, n_outputs=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    # inp = [torch.randn(1, 3, 225, 225).cuda() for i in range(4)]
    # out = model(inp)
    # print(out[0].shape, out[1].shape, out[2].shape)