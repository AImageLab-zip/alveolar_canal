import torch
import torch.nn as nn

def compute_and_normalize_coords(coords, MAX_X=168, MAX_Y=280, MAX_Z=360):
    max_dims = torch.tensor([MAX_X, MAX_Y, MAX_Z])-1
    patch_size = coords[:,3:] - coords[:,:3]

    # assert that all the elements are the same
    assert torch.all(patch_size[0] == patch_size), f"patch_size not constant across batch! Got: {patch_size}"

    patch_size = patch_size[0]
    dim_x, dim_y, dim_z = patch_size
    x = torch.arange(dim_x)
    y = torch.arange(dim_y)
    z = torch.arange(dim_z)

    cprod = torch.cartesian_prod(x,y,z).T
    cprod = cprod.reshape(3, dim_x, dim_y, dim_z)
    cprod = cprod.unsqueeze(0) # dim: 1 x 3 x X x Y x Z
    offset = coords[:, :3][:, :, None, None, None] # dim: B x 3 x 1 x 1 x 1
    max_dims = max_dims[None,:,None,None,None] # dim: 1 x 3 x 1 x 1 x 1
    return (cprod+offset)/max_dims

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class LatePosPadUNet3D(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch, size=32):
        self.n_classes = n_classes
        self.in_ch = in_ch
        super(LatePosPadUNet3D, self).__init__()

        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.ec0 = self.conv3Dblock(self.in_ch, size, groups=1)
        self.ec1 = self.conv3Dblock(size, size*2, kernel_size=3, padding=1, groups=1)  # third dimension to even val
        self.ec2 = self.conv3Dblock(size*2, size*2, groups=1)
        self.ec3 = self.conv3Dblock(size*2, size*4, groups=1)
        self.ec4 = self.conv3Dblock(size*4, size*4, groups=1)
        self.ec5 = self.conv3Dblock(size*4, size*8, groups=1)
        self.ec6 = self.conv3Dblock(size*8, size*8, groups=1)
        self.ec7 = self.conv3Dblock(size*8, size*16, groups=1)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = nn.ConvTranspose3d(size*16, size*16, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(size*8 + size*16, size*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(size*8, size*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(size*8, size*8, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(size*4 + size*8, size*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(size*4, size*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(size*4, size*4, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(size*2 + size*4, size*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(size*2, size*2, kernel_size=3, stride=1, padding=1)

        self.final = nn.ConvTranspose3d(size*2+3, n_classes, kernel_size=3, padding=1, stride=1)
        initialize_weights(self)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), groups=1, padding_mode='replicate'):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
                # nn.SiLU()
        )

    def forward(self, x, emb_codes):

        h = self.ec0(x)
        feat_0 = self.ec1(h)
        h = self.pool0(feat_0)
        h = self.ec2(h)
        feat_1 = self.ec3(h)

        h = self.pool1(feat_1)
        h = self.ec4(h)
        feat_2 = self.ec5(h)

        h = self.pool2(feat_2)
        h = self.ec6(h)
        h = self.ec7(h)

        # emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        # h = torch.cat((h, emb_pos), dim=1)
        h = torch.cat((self.dc9(h), feat_2), dim=1)

        h = self.dc8(h)
        h = self.dc7(h)

        h = torch.cat((self.dc6(h), feat_1), dim=1)
        h = self.dc5(h)
        h = self.dc4(h)

        h = torch.cat((self.dc3(h), feat_0), dim=1)
        h = self.dc2(h)
        h = self.dc1(h)

        coords = compute_and_normalize_coords(emb_codes)
        h = torch.cat((h, coords), dim=1)
        
        h = self.final(h)
        return torch.sigmoid(h)
