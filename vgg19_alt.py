
import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv2d_ds(nn.Module):
    def __init__(self, number_in, number_out):
        super(Conv2d_ds, self).__init__()
        self.depthwise = nn.Conv2d(
            number_in, number_in, kernel_size=3, padding=1, groups=number_in)
        self.pointwise = nn.Conv2d(number_in, number_out, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class VGG19_Alt(nn.Module):
    def __init__(self, use_classifier=False, number_of_classes=1000):
        super(VGG19_Alt, self).__init__()
        # vgg modules
        self.conv1_1 = Conv2d_ds(3, 64)
        self.conv1_2 = Conv2d_ds(64, 64)
        self.conv2_1 = Conv2d_ds(64, 128)
        self.conv2_2 = Conv2d_ds(128, 128)
        self.conv3_1 = Conv2d_ds(128, 256)
        self.conv3_2 = Conv2d_ds(256, 256)
        self.conv3_3 = Conv2d_ds(256, 256)
        self.conv3_4 = Conv2d_ds(256, 256)
        self.conv4_1 = Conv2d_ds(256, 512)
        self.conv4_2 = Conv2d_ds(512, 512)
        self.conv4_3 = Conv2d_ds(512, 512)
        self.conv4_4 = Conv2d_ds(512, 512)
        self.conv5_1 = Conv2d_ds(512, 512)
        self.conv5_2 = Conv2d_ds(512, 512)
        self.conv5_3 = Conv2d_ds(512, 512)
        self.conv5_4 = Conv2d_ds(512, 512)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.use_classifier = use_classifier

        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(16 * 24 * 512, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, number_of_classes),
            )

    def forward(self, x, out_keys=['r11', 'r21', 'r31', 'r41', 'r51', 'r42']):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])

        if self.use_classifier:
            x = out['p5']
            x = x.view(x.size(0), -1)
            return self.classifier(x)
        else:
            return [out[key] for key in out_keys]
