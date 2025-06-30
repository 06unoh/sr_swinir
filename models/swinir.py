import torch
from torch import nn

def window_partition(x, window_size):
  B, H, W, C=x.shape
  x=x.view(B, H//window_size, window_size, W//window_size, window_size, -1)
  windows=x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
  return windows

def window_reverse(window, window_size, H, W):
  B=int(window.shape[0]*window_size*window_size/(H*W))
  x=window.view(B, H//window_size, W//window_size, window_size, window_size, -1)
  x=x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
  return x

def get_attn_mask(window_size, shift_size, H, W, device):
  img_mask=torch.zeros((1, H, W, 1), device=device)
  cnt=0

  for h in (slice(-window_size),slice(-window_size,-shift_size),slice(-shift_size,None)):
    for w in (slice(-window_size),slice(-window_size,-shift_size),slice(-shift_size,None)):
      img_mask[:, h, w, :]=cnt
      cnt+=1
  img_window=window_partition(img_mask, window_size)
  img_window=img_window.view(img_window.shape[0], -1)
  attn_mask=img_window.unsqueeze(1)-img_window.unsqueeze(2)
  attn_mask=attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask==0, float(0.0))
  return attn_mask


class WindowAttention(nn.Module):
  def __init__(self, dim, num_heads, window_size):
    super().__init__()
    self.dim=dim
    self.n_heads=num_heads
    self.window_size=window_size
    self.scale=(dim/self.n_heads)**-0.5

    self.qkv=nn.Linear(dim, 3*dim)
    self.proj=nn.Linear(dim, dim)

  def forward(self, x, mask=None):
    B_, N, C=x.shape
    qkv=self.qkv(x).view(B_, N, 3, self.n_heads,C//self.n_heads)
    q, k, v=qkv.permute(2,0,3,1,4).contiguous()
    attn=(q@k.transpose(-2, -1))*self.scale

    if mask is not None:
      nW=mask.shape[0]
      attn=attn.view(B_//nW, nW, self.n_heads, N, N)
      attn=attn+mask.unsqueeze(1).unsqueeze(0)
      attn=attn.view(B_,self.n_heads,N,N)
    attn=attn.softmax(dim=-1)
    attn=(attn@v).permute(0,2,1,3).contiguous().view(B_, N, C)
    return self.proj(attn)

class DropPath(nn.Module):
  def __init__(self, drop_prob):
    super().__init__()
    self.drop_prob=drop_prob

  def forward(self, x):
    if self.drop_prob==0 or not self.training:
      return x

    keep_drop=1-self.drop_prob
    B=x.shape[0]
    shape=(B,)+(1,)*(x.ndim-1)
    random_tensor=torch.rand(shape, dtype=x.dtype, device=x.device)+keep_drop
    random_tensor.floor_()
    return x.div(keep_drop)*random_tensor

class SwinTFBlock(nn.Module):
  def __init__(self, dim, num_heads, window_size, shift_size ,drop_prob=0.1):
    super().__init__()
    self.window_size=window_size
    self.shift_size=shift_size

    self.norm1=nn.LayerNorm(dim)
    self.attn=WindowAttention(dim, num_heads, window_size)
    self.drop_path1=DropPath(drop_prob)

    self.norm2=nn.LayerNorm(dim)
    self.mlp=nn.Sequential(
        nn.Linear(dim, dim*4),
        nn.GELU(),
        nn.Linear(dim*4, dim)
    )
    self.drop_path2=DropPath(drop_prob)

  def forward(self, x, H, W):
    B, N, C=x.shape
    shortcut=x
    x=self.norm1(x).view(x.shape[0], H, W, -1)

    if self.shift_size>0:
      x=torch.roll(x, shifts=(-self.shift_size, -self.shift_size),dims=(1,2))

    x_windows=window_partition(x, self.window_size)
    x_windows=x_windows.view(-1, self.window_size*self.window_size, C)    #(B_, N, C)

    attn_mask=get_attn_mask(self.window_size, self.shift_size, H, W, x.device) if self.shift_size>0 else None
    x_attn=self.attn(x_windows, mask=attn_mask)   #(B_,N,C)

    x_attn=x_attn.view(-1,self.window_size, self.window_size, C)
    x=window_reverse(x_attn, self.window_size, H, W)   # (B, H, W ,C)

    if self.shift_size>0:
      x=torch.roll(x, shifts=(self.shift_size, self.shift_size),dims=(1,2))

    x=x.view(-1, H*W, C)
    x=shortcut+self.drop_path1(x)
    x=x+self.drop_path2(self.mlp(self.norm2(x)))
    return x

class RSTB(nn.Module):
  def __init__(self, dim, num_heads, window_size, drop_prob, depth):
    super().__init__()
    self.blocks=nn.ModuleList([
        SwinTFBlock(
            dim,
            num_heads,
            window_size,
            shift_size=0 if i%2==0 else window_size//2,
            drop_prob=0.1)
        for i in range(depth)
    ])
    self.conv=nn.Conv2d(dim, dim, 3, 1, 1)

  def forward(self, x):
    B, H, W, C=x.shape
    shortcut=x

    x=x.view(B,H*W,C)
    for blk in self.blocks:
      x=blk(x, H, W)
    x=x.view(B, H, W, C).permute(0,3,1,2).contiguous()
    x=self.conv(x)
    x=x.permute(0,2,3,1).contiguous()
    x=shortcut+x
    return x

class SwinIR(nn.Module):
  def __init__(self,img_dim=3, embed_dim=256, num_heads=8, window_size=8, drop_prob=0.1, depth=4, depths=3):
    super().__init__()
    self.conv_first=nn.Conv2d(img_dim, embed_dim, 3, 1, 1)
    self.layers=nn.ModuleList([
        RSTB(embed_dim, num_heads, window_size, drop_prob, depth)
        for _ in range(depths)
    ])
    self.norm=nn.LayerNorm(embed_dim)
    self.conv_after_body=nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    self.upsample=nn.Sequential(
        nn.Conv2d(embed_dim, embed_dim*4, 3, 1, 1),
        nn.PixelShuffle(2),
        nn.Conv2d(embed_dim, embed_dim*4, 3, 1, 1),
        nn.PixelShuffle(2),
        nn.Conv2d(embed_dim, img_dim, 3, 1, 1)
    )

  def forward(self, x):
    B, C, H, W=x.shape

    x=self.conv_first(x)
    x=x.permute(0,2,3,1).contiguous()

    for layer in self.layers:   # 좋은데 기본 트랜스포머 구조 (B,L,C)
      x=layer(x)

    x=self.norm(x)
    x=x.permute(0,3,1,2).contiguous()
    x=self.conv_after_body(x)
    return self.upsample(x)
