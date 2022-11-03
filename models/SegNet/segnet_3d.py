import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet3D(nn.Module):

    def __init__(self, num_classes, emb_shape, input_channels, BN_momentum=0.5):
        super(SegNet3D, self).__init__()
        self.in_chn = input_channels
        self.out_chn = num_classes
        emb_shape = (7,7,7)
        self.emb_shape = torch.as_tensor(emb_shape)

        # pos encoding
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())

        # encoding
        self.MaxEn = nn.MaxPool3d(2, stride=2, return_indices=True)

        self.ConvEn11 = nn.Conv3d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm3d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm3d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm3d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm3d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm3d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm3d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm3d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm3d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv3d(513, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm3d(512, momentum=BN_momentum)

        # decoding
        self.MaxDe = nn.MaxUnpool3d(2, stride=2)

        self.ConvDe53 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm3d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm3d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm3d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm3d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm3d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm3d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm3d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm3d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm3d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv3d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm3d(self.out_chn, momentum=BN_momentum)

    def forward(self, x, emb_codes):

        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        # append pos encoding
        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        x = torch.cat((x, emb_pos), dim=1)

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()


        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        return torch.sigmoid(x)

if __name__ == "__main__":
    model = SegNet3D(1, (8,8,8), 1)

    input = torch.rand((1,1,80,80,80))
    emb_codes = torch.rand((6))
    output = model(input, emb_codes)
