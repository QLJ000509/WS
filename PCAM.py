import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """


    B, C, H, W = input.size()
    rH = H // bin_size
    rW = W // bin_size
    out = input.view(B, C, bin_size, rH, bin_size, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B, -1, rH, rW, C)  # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size
    bin_num_w = bin_size
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous() # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out

class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out




class Net(network.resnet38d.Net):
    def __init__(self, bin_size, norm_layer):
        super().__init__()

        feat_inner = 4096 // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size**2
        self.gcn = GCN(bin_num, 4096)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(4096, feat_inner)
        self.proj_key = nn.Linear(4096, feat_inner)
        self.proj_value = nn.Linear(4096, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, 4096, kernel_size=1, bias=False),
            norm_layer(4096),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)

        # self.convb = nn.Sequential(
        #     nn.Conv2d(4096, 4096, kernel_size=3, padding=1, bias=False),
        #     norm_layer(4096),
        #     nn.ReLU(inplace=True)
        # )
        # seg_convs = [
        #     nn.Conv2d(8192, 4096, kernel_size=3, padding=1, bias=False),
        #     norm_layer(4096),
        #     nn.ReLU(inplace=True)
        # ]
        # seg_convs.append(nn.Conv2d(4096, 21, kernel_size=1))
        # self.conv_seg = nn.Sequential(*seg_convs)


        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight) #权重初始化
        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)
        self.fc8_seg_conv2 = nn.Conv2d(512, 21, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

        # self.not_training = [self.conv1a]
        self.not_training = []
        # self.from_scratch_layers = []
        self.from_scratch_layers = [self.fc8, self.fc8_seg_conv1, self.fc8_seg_conv2]


    def forward(self, x, require_seg = True, require_mcam = True):
        x = super().forward(x)  #(1,4096,41,41)
        residual = x
        if require_seg == True and require_mcam == True:
            x_cam = x.clone()
            x_feats = x.clone()

            x = self.dropout7(x)      #(1,4096,41,41)
            x = F.avg_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=0)  #(1,4096,1,1)
            x = self.fc8(x)             #(1,20,1,1)
            x = x.view(x.size(0), -1)   #(1,20) reshape操作：（batch size,channel,H,W）->(batch size,channel*H*W)  以batchsize为行数展平

            cam = F.conv2d(x_cam, self.fc8.weight)    #(1,4096,41,41)*(20,4096,1,1)->(1,20,41,41)
            cam = F.relu(cam)                         #(1,20,41,41)
            cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]
            pcam = patch_split(cam, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, K]
            x_feats = patch_split(x_feats, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, C]

            B = pcam.shape[0]
            rH = pcam.shape[2]
            rW = pcam.shape[3]
            K = pcam.shape[-1]
            C = x_feats.shape[-1]
            pcam = pcam.view(B, -1, rH * rW, K)  # [B, bin_num_h * bin_num_w, rH * rW, K]
            x_feats = x_feats.view(B, -1, rH * rW, C)  # [B, bin_num_h * bin_num_w, rH * rW, C]

            bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)  # [B, bin_num_h * bin_num_w, K, 1]
            pixel_confidence = F.softmax(pcam, dim=2)

            local_feats = torch.matmul(pixel_confidence.transpose(2, 3),
                                       x_feats) * bin_confidence  # [B, bin_num_h * bin_num_w, K, C]
            local_feats = self.gcn(local_feats)  # [B, bin_num_h * bin_num_w, K, C]
            global_feats = self.fuse(local_feats)  # [B, 1, K, C]
            global_feats = self.relu(global_feats).repeat(1, x_feats.shape[1], 1, 1)  # [B, bin_num_h * bin_num_w, K, C]

            query = self.proj_query(x_feats)  # [B, bin_num_h * bin_num_w, rH * rW, C//2]
            key = self.proj_key(local_feats)  # [B, bin_num_h * bin_num_w, K, C//2]
            value = self.proj_value(global_feats)  # [B, bin_num_h * bin_num_w, K, C//2]

            aff_map = torch.matmul(query, key.transpose(2, 3))  # [B, bin_num_h * bin_num_w, rH * rW, K]
            aff_map = F.softmax(aff_map, dim=-1)
            out = torch.matmul(aff_map, value)  # [B, bin_num_h * bin_num_w, rH * rW, C]

            out = out.view(B, -1, rH, rW, value.shape[-1])  # [B, bin_num_h * bin_num_w, rH, rW, C]
            out = patch_recover(out, self.bin_size)  # [B, C, H, W]

            out = residual + self.conv_out(out)

            x_seg = self.fc8_seg_conv1(out)         #(1,512,41,41)
            x_seg = F.relu(x_seg)                     #(1,512,41,41)
            x_seg = self.fc8_seg_conv2(x_seg)         #(1,21,41,41)
            # seg = self.convb(out)
            # x_seg = self.conv_seg(torch.cat([seg, residual], dim=1))

            return x, cam, x_seg
        elif require_mcam == True and require_seg== False:
            x_cam = x.clone()
            cam = F.conv2d(x_cam, self.fc8.weight)
            cam = F.relu(cam)

            return cam

        else:
            x = self.dropout7(x)

            x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

            x = self.fc8(x)
            x = x.view(x.size(0), -1)

            return x

    def forward_cam(self, x):
        x = super().forward(x)

        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class SegNet(Net):
    def __init__(self):
        super().__init__(bin_size=4, norm_layer=nn.BatchNorm2d)

    def forward(self, x, require_seg=True, require_mcam= True):
        if require_seg == True and require_mcam == True:
            input_size_h = x.size()[2]         #448
            input_size_w = x.size()[3]         #448
            # self.interp1 = nn.UpsamplingBilinear2d(size=(int(input_size_h * 0.5), int(input_size_w * 0.5)))
            # self.interp2 = nn.UpsamplingBilinear2d(size=(int(input_size_h * 1.5), int(input_size_w * 1.5)))
            # self.interp3 = nn.UpsamplingBilinear2d(size=(int(input_size_h * 2), int(input_size_w * 2)))
            # x2 = self.interp1(x)
            # x3 = self.interp2(x)
            # x4 = self.interp3(x)
            x2 = F.interpolate(x, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear',align_corners=False)
            x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear',align_corners=False)
            x4 = F.interpolate(x, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear',align_corners=False)
            #(1,3,224,224)    (1,3,672,672)    (1,3,896,896)
            seg = []
            with torch.enable_grad():
                xf1, cam1, seg1 = super().forward(x,require_seg=True, require_mcam=True)
            with torch.no_grad():
                cam2 = super().forward(x2,require_seg=False, require_mcam=True)  #(1,20,28,28)
                cam3 = super().forward(x3,require_seg=False, require_mcam=True)  #(1,20,84,84)
                cam4 = super().forward(x4,require_seg=False, require_mcam=True)  #(1,20,112,112)

            xf_temp = xf1

            cam2 = F.interpolate(cam2, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear',align_corners=False)#(1,20,56,56)
            cam3 = F.interpolate(cam3, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear',align_corners=False)#(1,20,56,56)
            cam4 = F.interpolate(cam4, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear',align_corners=False)#(1,20,56,56)
            # self.interp_final = nn.UpsamplingBilinear2d(size=(int(seg1.shape[2]), int(seg1.shape[3])))
            # cam2 = self.interp_final(cam2)
            # cam3 = self.interp_final(cam3)
            # cam4 = self.interp_final(cam4)

            cam = (cam1+cam2+cam3+cam4)/4   #(1,20,56,56)

            seg.append(seg1)  # for original scale

            return xf_temp, cam, seg

        if require_mcam == False and require_seg == False:
            xf = super().forward(x,require_seg=False,require_mcam=False)
            self.not_training = [self.conv1a]
            return xf
        if require_mcam == False and require_seg == True:
            xf, cam, seg = super().forward(x, require_seg=True, require_mcam=True)
            return seg



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

if __name__ == "__main__":
 x = torch.randn(1, 3, 320, 320)
 model =SegNet()
 model.cuda()
 out=model(x.cuda())
 # print(out.shape)