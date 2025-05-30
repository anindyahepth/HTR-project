import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, Height, Width, dropout=0.):
        super().__init__()
        self.Lin1 = nn.Linear(dim,hidden_dim)
        self.act = nn.GELU()
        self.BN = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.Lin2 = nn.Linear(hidden_dim, dim)
        self.height = Height
        self.width = Width

    def forward(self, x):
        x = self.Lin1(x)
        b, n, d = x.shape
        h = self.height
        w = self.width
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        x = self.BN(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.act(x)
        x = self.drop(x)
        x = self.Lin2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, Height, Width, l_max =3, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.gelu = nn.GELU()
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([])
        self.height = Height
        self.width = Width
        self.l_max = l_max

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, Height, Width, dropout=dropout))
            ]))

            self.convs.append(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim))
            self.batchnorms.append(nn.BatchNorm2d(dim))

    def forward(self, x):
        for i, [attn, ff] in enumerate(self.layers):
            
            if i < self.l_max:
              shortcut = x
              b, n, d = shortcut.shape
              h = self.height
              w = self.width
              shortcut = rearrange(shortcut, 'b (h w) d -> b d h w', h=h, w=w) 
              shortcut = self.gelu(shortcut)
              shortcut = self.batchnorms[i](shortcut)
              shortcut = self.convs[i](shortcut)
              shortcut = rearrange(shortcut, 'b d h w -> b (h w) d')
            
            else:
              shortcut = torch.zeros_like(x)

            x = attn(x) + x
            x = ff(x) + x
            x = shortcut + x
              
              
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=64, l_max, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        Height = image_height // patch_height
        Width = image_width // patch_width
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width, c=channels),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, Height, Width, l_max, dropout)

        
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        

        x = self.to_latent(x)
        return self.mlp_head(x)
