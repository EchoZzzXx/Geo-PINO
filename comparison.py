import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import os
import sys 
sys.path.append('/mnt/pool/zx/Geo-PINO')
print('000')
from utilities3 import *

from elas_geofno import FNO2d as GeoFNO2d
from elas_geofno import IPHI as GeoIPHI


##########################################################
# 为FNO写的类
#########################################################




##########################################################
#为PINO写的类
##########################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y

def elasticity_pde_loss_autograd(mesh, sigma):
    print("Input shapes:", mesh.shape, sigma.shape)
    print("Input stats - mesh:", torch.min(mesh).item(), torch.max(mesh).item())
    print("Input stats - sigma:", torch.min(sigma).item(), torch.max(sigma).item())
    
    mesh.requires_grad_(True)
    
    print("Computing gradients...")
    # 计算梯度前检查
    if torch.isnan(sigma).any():
        print("NaN already in sigma before gradient")
        return torch.tensor(0.0, device=mesh.device, requires_grad=True)
    
    # 计算一小部分样本点的梯度进行测试
    test_sigma = sigma[:1, :10, :].contiguous()
    try:
        test_grad = grad(outputs=test_sigma.sum(), inputs=mesh[:1, :10, :].contiguous(), 
                    create_graph=False, retain_graph=False)[0]
        print("Test gradient successful:", test_grad.shape)
    except Exception as e:
        print(f"Test gradient failed: {str(e)}")  #问题出在这一步
    
    # 正常计算
    try:
        sigma_prime = grad(outputs=sigma.sum(), inputs=mesh, 
                          create_graph=True, retain_graph=True)[0]
        print("Gradient stats:", torch.min(sigma_prime).item(), torch.max(sigma_prime).item())
        
        div_sigma = sigma_prime.sum(dim=-1, keepdim=True)
        print("Divergence stats:", torch.min(div_sigma).item(), torch.max(div_sigma).item())
        
        # 平衡方程要求div(σ) = 0
        target = torch.zeros_like(div_sigma)
        
        # 使用LpLoss计算损失会出现NaN，暂未解决
        norm_div = torch.norm(div_sigma, p=2)
        norm_target = torch.norm(target, p=2) + 1e-10
        loss = norm_div / norm_target
        print("Loss value:", loss.item())
        
        return loss
    except Exception as e:
        print(f"Exception during gradient computation: {str(e)}")
        import traceback
        traceback.print_exc()
        return torch.tensor(0.0, device=mesh.device, requires_grad=True)



def elasticity_pde_loss_discrete(mesh, sigma, k=5, sample_size=45):
    """
    高效的PDE损失计算函数，每个批次只采样固定数量的点
    
    Args:
        mesh: [batch_size, mesh_num, 2] 网格坐标
        sigma: [batch_size, mesh_num, 1] 应力场
        k: 近邻数量，默认5个
        sample_size: 每个批次中采样的点数，默认45个
    
    Returns:
        loss: 散度的均方误差损失
    """
    batch_size = mesh.shape[0]
    mesh_num = mesh.shape[1]
    device = mesh.device
    
    # 预分配批次结果存储空间
    div_sigma_all = torch.zeros(batch_size, sample_size, 1, device=device)
    
    for b in range(batch_size):
        # 随机选择
        indices = torch.randperm(mesh_num, device=device)[:sample_size]
        
        # 提取选中的点和应力值
        sample_points = mesh[b, indices]  # [sample_size, 2]
        sample_stress = sigma[b, indices, 0]  # [sample_size]
        
        # 为每个样本点找到k个最近邻(不包括自身)
        dist = torch.cdist(sample_points, mesh[b])  # [sample_size, mesh_num]
        
        # 将自己与自己的距离设为无穷大
        for i in range(sample_size):
            dist[i, indices[i]] = float('inf')
        
        # 获取最近的k个点的索引
        _, nn_indices = torch.topk(dist, k=k, largest=False)  # [sample_size, k]
        
        # 批量计算散度
        for i in range(sample_size):
            # 获取邻居点的坐标和应力
            neighbor_points = mesh[b, nn_indices[i]]  # [k, 2]
            neighbor_stress = sigma[b, nn_indices[i], 0]  # [k]
            
            # 计算坐标差值
            dx = neighbor_points[:, 0] - sample_points[i, 0]  # [k]
            dy = neighbor_points[:, 1] - sample_points[i, 1]  # [k]
            
            # 计算应力差值
            ds = neighbor_stress - sample_stress[i]  # [k]
            
            # 构建简化版最小二乘问题
            try:
                # 为每个方向计算加权平均导数
                weights = 1.0 / (torch.sqrt(dx**2 + dy**2) + 1e-8)
                
                # 计算x方向导数 (使用投影)
                x_weights = weights * dx**2 / (dx**2 + dy**2 + 1e-8)
                grad_x = torch.sum(ds * x_weights * torch.sign(dx)) / (torch.sum(x_weights) + 1e-8)
                
                # 计算y方向导数 (使用投影)
                y_weights = weights * dy**2 / (dx**2 + dy**2 + 1e-8)
                grad_y = torch.sum(ds * y_weights * torch.sign(dy)) / (torch.sum(y_weights) + 1e-8)
                
                # 散度是两个方向导数的和
                div_sigma_all[b, i, 0] = grad_x + grad_y
                
            except:
                # 若计算出错，则赋予零值
                div_sigma_all[b, i, 0] = 0.0
    
    # 弹性平衡方程要求散度为零
    target = torch.zeros_like(div_sigma_all)
    loss = torch.nn.MSELoss()(div_sigma_all, target)
    
    return loss



class ElasticityEqnLoss(object):
    def __init__(self):
        super().__init__()
    
    def discrete(self, mesh, sigma):
        """
        使用数值离散方法计算弹性方程的PDE损失
        """
        return elasticity_pde_loss_discrete(mesh, sigma)
    
    def __call__(self, mesh, sigma):
        """
        计算PDE损失的主入口
        """
        return self.discrete(mesh, sigma)



class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels, is_mesh=True, s1=40, s2=40):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)


        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u
    

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()

        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3*self.width, 4*self.width)
        self.fc1 = nn.Linear(4*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, 4*self.width)
        self.fc4 = nn.Linear(4*self.width, 2)
        self.activation = torch.tanh
        self.center = torch.tensor([0.5, 0.5], device="cuda").reshape(1,1,2)  # 修改为单元格中心

        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)


    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


print('0 completed')

# 设置路径
PATH = '.'
# 模型保存路径
geofno_model_path = PATH + '/model/elas/elas_geofno_mse_311/ckpt'
geopino_model_path = PATH + '/model/elas/elas_pino-310/ckpt'
# 可视化结果保存路径
vis_path = PATH + '/model/elas/model_comparison——new/'
if not os.path.exists(vis_path):
    os.makedirs(vis_path)

# 加载数据
PATH_Sigma = PATH+'/data/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
PATH_XY = PATH+'/data/elasticity/Meshes/Random_UnitCell_XY_10.npy'
PATH_rr = PATH+'/data/elasticity/Meshes/Random_UnitCell_rr_10.npy'

input_rr = np.load(PATH_rr)
input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0)
input_s = np.load(PATH_Sigma)
input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
input_xy = np.load(PATH_XY)
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)

# 设置参数
ntrain = 1000
ntest = 200
batch_size = 20
modes = 12
width = 32

# 准备测试数据
test_rr = input_rr[-ntest:]
test_s = input_s[-ntest:]
test_xy = input_xy[-ntest:]

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_rr, test_s, test_xy), 
                                          batch_size=batch_size, shuffle=False)


model_fno = GeoFNO2d(modes, modes, width, in_channels=2, out_channels=1).cuda()
model_fno.eval()
model_fno_iphi = GeoIPHI().cuda()
model_fno_iphi.eval()

model_pino = FNO2d(modes, modes, width, in_channels=2, out_channels=1).cuda()
model_pino.eval()
model_pino_iphi = IPHI().cuda()
model_pino_iphi.eval()

myloss = nn.MSELoss()

print('1 completed')

def load_checkpoint(checkpoint_path):
    """加载模型检查点"""
    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    else:
        print(f"检查点不存在: {checkpoint_path}")
        return None

def compare_models(epoch):
    """加载特定epoch的两个模型并进行对比"""
    # 加载GeoFNO模型
    geofno_checkpoint_path = os.path.join(geofno_model_path, f'checkpoint_ep{epoch}.pth')
    geofno_checkpoint = load_checkpoint(geofno_checkpoint_path)
    
    # 加载GeoPINO模型
    geopino_checkpoint_path = os.path.join(geopino_model_path, f'checkpoint_ep{epoch}.pth')
    geopino_checkpoint = load_checkpoint(geopino_checkpoint_path)
    
    if geofno_checkpoint is None or geopino_checkpoint is None:
        print(f"跳过Epoch {epoch}，未能同时找到两个模型的检查点")
        return

    model_fno.load_state_dict(geofno_checkpoint['model_state_dict'])
    model_fno_iphi.load_state_dict(geofno_checkpoint['iphi_state_dict'])

    model_pino.load_state_dict(geopino_checkpoint['model_state_dict'])
    model_pino_iphi.load_state_dict(geopino_checkpoint['iphi_state_dict'])


    print(f'对比Epoch{epoch}的模型性能')


    # 为这个epoch创建专门的目录
    epoch_dir = os.path.join(vis_path, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    with torch.no_grad():
        for rr, sigma, mesh in test_loader:
            rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
            outfno = model_fno(mesh, code=rr, iphi=model_fno_iphi)
            outpino = model_pino(mesh, code=rr, iphi=model_pino_iphi)

    XY = mesh[-1].squeeze().detach().cpu().numpy()
    truth = sigma[-1].squeeze().detach().cpu().numpy()
    pred_fno = outfno[-1].squeeze().detach().cpu().numpy()
    pred_pino = outpino[-1].squeeze().detach().cpu().numpy()
    error_fno = truth - pred_fno
    error_pino = truth - pred_pino
    emin = min(error_fno.min(),error_pino.min())
    emax = max(error_fno.max(),error_pino.max())


    lims  = dict(cmap='RdBu_r', vmin=truth.min(), vmax=truth.max())
    errors = dict(cmap = 'viridis', vmin=emin, vmax=emax)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    ax[0,0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
    ax[1,0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
    ax[0,1].scatter(XY[:, 0], XY[:, 1], 100, pred_fno, edgecolor='w', lw=0.1, **lims)
    ax[1,1].scatter(XY[:, 0], XY[:, 1], 100, pred_pino, edgecolor='w', lw=0.1, **lims)
    ax[0,2].scatter(XY[:, 0], XY[:, 1], 100, error_fno, edgecolor='w', lw=0.1, **errors)
    ax[1,2].scatter(XY[:, 0], XY[:, 1], 100, error_pino, edgecolor='w', lw=0.1, **errors)

    ax[0,0].set_title('ground truth')
    ax[1,0].set_title('ground truth')
    ax[0,1].set_title('GeoFNO pred')
    ax[1,1].set_title('GeoPINO pred')
    ax[0,2].set_title('GeoFNO error')
    ax[1,2].set_title('GeoPINO error')

    file_path = os.path.join(epoch_dir,'ep'+str(epoch)+'.png')
    fig.savefig(file_path, dpi=300)

    print(f"已完成Epoch {epoch}的可视化，结果保存在: {epoch_dir}")


# 要比较的epoch列表
epochs_to_compare = [0, 1, 10]  # 按照您的要求

# 依次处理每个epoch
for epoch in epochs_to_compare:
    compare_models(epoch)

print("\n所有模型对比完成，结果保存在:", vis_path)
