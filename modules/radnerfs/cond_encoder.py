import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatMapEncoder(nn.Module):
    def __init__(self, dim_in=768, dim_aud=64, win_size=16):
        super(HeatMapEncoder, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud

        num_layers = 5
        kernel_size = 3
        dilation_rate = 2

        lst = []
        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            if i == 0:
                in_dim = dim_in
                out_dim = 16
            else:
                in_dim = 16
                out_dim = 16
            lst.append(nn.Conv2d(in_dim, out_dim, kernel_size, dilation=dilation, padding=padding))
            lst.append(nn.LeakyReLU(0.02, True))
        self.conv = nn.Sequential(*lst)

        self.fc = nn.Sequential(
            nn.Linear(dim_in, 32),
            # nn.Linear(16, 32),
            nn.LeakyReLU(0.02, True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.02, True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.02, True),
            # nn.Linear(64, 64),
            # nn.LeakyReLU(0.02, True),
            nn.Linear(32, dim_aud),
        )
    def forward(self, x):
        """
        x: [b, 1, n_rays, c]
        """
        x = x.squeeze() # [b,t,c]

        # uncomment to use cnn
        # x = x.transpose(1,2) # [b,c,t]
        # n_rays = x.shape[-1]
        # b = x.shape[0]
        # w = int(n_rays**0.5)
        # x = self.conv(x.reshape([b,-1,w,w])).reshape([b,-1,n_rays]).transpose(1,2) # [b,c,t]=>[b,t,c]
        
        x = self.fc(x)
        return  x# [b, n_rays, c]


class HeatMapAttNet(nn.Module):
    # Audio feature attention-based smoother in AD-NeRF
    def __init__(self, in_out_dim=64, seq_len=5):
        super(HeatMapAttNet, self).__init__()
        self.seq_len = seq_len
        self.in_out_dim = in_out_dim
        self.attention_fc = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Linear(self.in_out_dim, 16),
            nn.LeakyReLU(0.02, True),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.02, True),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.02, True),
            nn.Linear(4, 2),
            nn.LeakyReLU(0.02, True),
            nn.Linear(2, 1),
            nn.LeakyReLU(0.02, True),
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        x: [b=8, n_rays, c]
        return:
            [n_rays, c]
        """
        y = self.attention_fc(x).squeeze() # [b,n,c] ==> [b, n, 1] ==> [b,n]
        y = y.permute(1,0) # [b,n] ==> [n,b]
        y = self.attentionNet(y).permute(1,0).unsqueeze(2) # [n, b] ==> [b, n] ==> [b, n, 1]
        smoothed_y = torch.sum(y*x, dim=0) # [b, n, 1] * [b, n, c] ==> [n, c]
        return smoothed_y # [n_rays, c]
    

# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        if win_size == 1:
            strides = [1,1,1,1]
        elif win_size == 2:
            strides = [2,1,1,1]
        elif win_size in [3, 4]:
            strides = [2,2,1,1]
        elif win_size == [5, 8]:
            strides = [2,2,2,1]
        elif win_size == 16:
            strides = [2,2,2,2]
        else:
            raise ValueError("unsupported win_size")
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=strides[0],
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=strides[1],
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=strides[2],
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=strides[3],
                      padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        """
        x: [b, t_window, c]
        """
        half_w = int(self.win_size/2)
        x = x.permute(0, 2, 1) # [b,t=16,c]=>[b,c,t=16]
        x = self.encoder_conv(x).squeeze(-1) # [b, c=64, 1] => [b, c]
        x = self.encoder_fc1(x) # .squeeze() # [b,out_dim=76]
        return x


class AudioAttNet(nn.Module):
    # Audio feature attention-based smoother in AD-NeRF
    def __init__(self, in_out_dim=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.in_out_dim = in_out_dim
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.in_out_dim, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        x: [b=8, c]
        return:
            [c]
        """
        y = x[:, :self.in_out_dim].permute(1, 0).unsqueeze(0)  # [b, c] => [1, c, b]
        y = self.attentionConvNet(y) # [1,1,b]
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1) # [8, 1]
        smoothed_y = torch.sum(y*x, dim=0) # [8,1]*[8,c]=>[8,c]=>[c,]
        return smoothed_y
    

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x