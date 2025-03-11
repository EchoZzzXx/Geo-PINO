"""
@author: Zongyi Li and Daniel Zhengyu Huang
Modified to include PINO (Physics-Informed Neural Operator) for elasticity problems
"""

import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import os
import sys 
sys.path.append('/ailab/user/zhangjinouwen/zuoxiang/Geo-FNO')
from utilities3 import *
from Adam import Adam
from torch.autograd import grad
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# fourier layer
################################################################
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


################################################################
# configs
################################################################
Ntotal = 2000
ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 11
step_size = 50
gamma = 0.5

modes = 12
width = 32

################################################################
# load data and data normalization
################################################################
PATH = '.'
PATH_Sigma = PATH+'/data/elasticity/Meshes/Random_UnitCell_sigma_10.npy' # 应力场
PATH_XY = PATH+'/data/elasticity/Meshes/Random_UnitCell_XY_10.npy'   # 网格坐标
PATH_rr = PATH+'/data/elasticity/Meshes/Random_UnitCell_rr_10.npy'   #在IPHI里有用，但物理意义是？
# PATH_theta = PATH+'/data/elasticity/Meshes/Random_UnitCell_theta_10.npy' #这是什么东西？ 旋转角度？ 但是这个角度是如何定义的？

input_rr = np.load(PATH_rr)   # 参数或微观结构（？）
input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0)
input_s = np.load(PATH_Sigma)  # 应力场
input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
input_xy = np.load(PATH_XY)  # 网格坐标
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)
# input_theta = np.load(PATH_theta)  

train_rr = input_rr[:ntrain]
test_rr = input_rr[-ntest:]
train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]

print(input_rr.shape, input_s.shape, input_xy.shape)
# print(input_theta.shape)
# print(input_rr[1,:],input_rr[1,:].shape)
# print(input_s[1,:,:],input_s[1,:,:].shape)
# print(input_xy[1,:,:],input_xy[1,:,:].shape)

# print(train_rr.shape, train_s.shape, train_xy.shape)
# print(test_rr.shape, test_s.shape, test_xy.shape)



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_rr, train_s, train_xy), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_rr, test_s, test_xy), batch_size=batch_size,
                                          shuffle=False)




################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width, in_channels=2, out_channels=1).cuda()
model_iphi = IPHI().cuda()
print(count_params(model), count_params(model_iphi))

params = list(model.parameters()) + list(model_iphi.parameters())
optimizer = Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# myloss = LpLoss(size_average=False)
myloss = nn.MSELoss()
N_sample = 1000

save_path = PATH + '/model/elas/elas_pino-310/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_save_path = PATH +  '/model/elas/elas_pino-310/ckpt'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 物理损失权重
physics_weight = 0.1
pde_loss = ElasticityEqnLoss()

train_data_losses = []
train_phy_losses = []
test_losses = []
epochs_list = []

# def plot_losses(epochs, train_data, train_phy, test, save_path):
#     """
#     绘制训练过程中的各种损失曲线
    
#     Args:
#         epochs: epoch列表
#         train_data: 训练数据损失
#         train_phy: 训练物理损失
#         test: 测试损失
#         save_path: 保存路径
#     """
#     plt.figure(figsize=(24, 18))
    
#     # 绘制训练数据损失和测试损失
#     plt.subplot(2, 2, 1)
#     plt.semilogy(epochs, train_data, 'b-', label='Train Data Loss')
#     plt.semilogy(epochs, test, 'r-', label='Test Loss')
    
#     # 要标注的索引位置：前两个点和最后一个点
#     indices_to_annotate = [0, 1, len(epochs)-1] if len(epochs) > 2 else range(len(epochs))
    
#     # 标出选定点的纵坐标值
#     for i in indices_to_annotate:
#         plt.annotate(f'{train_data[i]:.4f}', 
#                     xy=(epochs[i], train_data[i]), 
#                     xytext=(0, 10), 
#                     textcoords='offset points',
#                     color='blue',
#                     ha='center')
        
#         plt.annotate(f'{test[i]:.4f}', 
#                     xy=(epochs[i], test[i]), 
#                     xytext=(0, -20), 
#                     textcoords='offset points',
#                     color='red',
#                     ha='center')
    
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss (log scale)')
#     plt.title('Data Loss')
#     plt.legend()
#     plt.grid(True, which="both", ls="--")
    
#     # 绘制物理损失
#     plt.subplot(2, 2, 2)
#     plt.semilogy(epochs, train_phy, 'g-', label='Physics Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss (log scale)')
#     plt.title('Physics Loss')
#     plt.legend()
#     plt.grid(True, which="both", ls="--")
    
#     # 绘制所有损失的比较
#     plt.subplot(2, 2, 3)
#     plt.semilogy(epochs, train_data, 'b-', label='Train Data')
#     plt.semilogy(epochs, train_phy, 'g-', label='Physics')
#     plt.semilogy(epochs, test, 'r-', label='Test')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss (log scale)')
#     plt.title('All Losses Comparison')
#     plt.legend()
#     plt.grid(True, which="both", ls="--")
    
#     plt.tight_layout()
    
#     # 保存图像
#     loss_path = os.path.join(save_path, 'losses_500.png')
#     plt.savefig(loss_path, dpi=300)
#     plt.close()

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    train_phy = 0
    
    for rr, sigma, mesh in train_loader:
        rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
        samples_x = torch.rand(batch_size, N_sample, 2).cuda() * 3 -1
        # samples_x = torch.rand(batch_size, N_sample, 2).cuda()  # 在[0,1]x[0,1]范围内采样
        
        optimizer.zero_grad()
        
        # # 需要启用梯度计算以计算物理损失
        # mesh.requires_grad_(True)
        
        out = model(mesh, code=rr, iphi=model_iphi)
        samples_xi = model_iphi(samples_x, code=rr)

        # 数据损失
        loss_data = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
        # print(f'1111111' + str(loss_data))
        
        # 正则化损失
        loss_reg = myloss(samples_xi, samples_x)
        # print(f'2222222' + str(loss_reg))
        
        # PINO physics loss
        loss_physics = pde_loss(mesh, out)
        
        # 总损失
        loss = loss_data + 0.000 * loss_reg + physics_weight * loss_physics
        loss.backward()

        # print(f"Epoch {ep}, Batch Gradient Norms:")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"  {name}: {grad_norm}")
        #     else:
        #         print(f"  {name}: No gradient")

        optimizer.step()
        train_l2 += loss_data.item()
        train_reg += loss_reg.item()
        train_phy += loss_physics.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, sigma, mesh in test_loader:
            rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
            out = model(mesh, code=rr, iphi=model_iphi)
            test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()

    train_l2 /= ntrain
    train_reg /= ntrain
    train_phy /= ntrain
    test_l2 /= ntest

    train_data_losses.append(train_l2)
    train_phy_losses.append(train_phy)
    test_losses.append(test_l2)
    epochs_list.append(ep)


    t2 = default_timer()
    print(ep, t2 - t1, train_l2, train_reg, train_phy, test_l2)
    # print(f"@@@@@@@{out[-1]}@@@@@@@")

    if ep < 5 or (ep >= 10 and ep % 10 == 0):

        # 保存模型状态
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'iphi_state_dict': model_iphi.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_l2,
            'test_loss': test_l2,
        }, os.path.join(model_save_path, f'checkpoint_ep{ep}.pth'))
        
        print(f"Model saved at epoch {ep}")

    # if ep >= 300 and ep % 20 == 0:
    #     plot_losses(epochs_list, train_data_losses, train_phy_losses, test_losses, save_path)


    
    XY = mesh[-1].squeeze().detach().cpu().numpy()
    truth = sigma[-1].squeeze().detach().cpu().numpy()
    pred = out[-1].squeeze().detach().cpu().numpy()
    error = truth - pred

    # 创建图形，适当调整比例
    fig = plt.figure(figsize=(16, 5))
    
    # 创建三个子图区域，并留出colorbar的空间
    # 前两个子图放在左边，第三个子图放在右边
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 0.1, 1.3])  # 4列，比例为1:1:0.1:1.3
    ax1 = plt.subplot(gs[0, 0])  # 第一个子图 (truth)
    ax2 = plt.subplot(gs[0, 1])  # 第二个子图 (pred)
    ax_cbar1 = plt.subplot(gs[0, 2])  # 第一个colorbar区域
    ax3 = plt.subplot(gs[0, 3])  # 第三个子图 (error)
    
    # 设置统一的颜色范围
    vmin_data = truth.min()
    vmax_data = truth.max()
    vmin_error = error.min()
    vmax_error = error.max()
    
    # 绘制散点图
    sc0 = ax1.scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, 
                      cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data)
    sc1 = ax2.scatter(XY[:, 0], XY[:, 1], 100, pred, edgecolor='w', lw=0.1, 
                      cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data)
    sc2 = ax3.scatter(XY[:, 0], XY[:, 1], 100, error, edgecolor='w', lw=0.1, 
                      cmap='RdBu_r', vmin=vmin_error, vmax=vmax_error)

    # 设置标题
    ax1.set_title('Ground Truth', fontsize=12)
    ax2.set_title('PINO Prediction', fontsize=12)
    ax3.set_title('Error', fontsize=12)
    
    # 添加colorbar
    # 第一个colorbar放在前两张图中间
    cbar1 = plt.colorbar(sc0, cax=ax_cbar1)
    cbar1.set_label('Value', fontsize=10)
    
    # 第二个colorbar放在第三张图右边
    divider = make_axes_locatable(ax3)  # 需要导入 from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(sc2, cax=cax2)
    cbar2.set_label('Error', fontsize=10)
    
    # 添加全局标题
    plt.suptitle(f'Epoch {ep}', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    file_name = 'ep' + str(ep) + '.png'
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，释放内存
