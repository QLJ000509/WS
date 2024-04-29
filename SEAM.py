import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)

        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, 21, (3, 3), stride=1, padding=12, dilation=12, bias=True)

        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2, self.fc8_seg_conv1, self.fc8_seg_conv2]


    def forward(self, x, require_seg = True, require_mcam = True):

      d = super().forward_as_dict(x)

      if require_seg == True and require_mcam == True:

        x_seg = self.fc8_seg_conv1(self.dropout7(d['conv6']))
        x_seg = F.relu(x_seg)
        x_seg = self.fc8_seg_conv2(x_seg)

        cam = self.fc8(self.dropout7(d['conv6']))

        k = super().forward(x)

        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0


        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)

        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)

        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        cam = self.PCM(cam_d_norm, f)

        k = self.dropout7(k)
        k = F.avg_pool2d(k, kernel_size=(k.size(2), k.size(3)), padding=0)
        k = self.fc8(k)
        k = k.view(k.size(0), -1)

        return k, cam, x_seg

      elif require_mcam == True and require_seg == False:
          x = super().forward(x)
          x_cam = x.clone()
          cam = F.conv2d(x_cam, self.fc8.weight)
          cam = F.relu(cam)

          return cam

      else:
          x = super().forward(x)
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

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

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
        super().__init__()

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
 x = torch.randn(1, 3, 321, 321)
 model =SegNet()
 model.cuda()
 out=model(x.cuda())
 # print(out.shape)
 # flops, params = profile(model, (x,))
 # flops, params = clever_format([flops, params], "%.3f")
 # print("FLOPs: %s" % (flops))
 # print("params: %s" % (params))