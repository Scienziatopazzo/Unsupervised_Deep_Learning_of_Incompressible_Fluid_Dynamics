import torch
import torch.nn.functional as F
import math
import get_param

params = get_param.params()

current_cuda = 0

def toCuda(x, cuda_i=None):
	if cuda_i is None:
		cuda_i = current_cuda
	cuda_i = int(cuda_i)
	if type(x) is tuple:
		return [xi.cuda(cuda_i) if params.cuda else xi for xi in x]
	return x.cuda(cuda_i) if params.cuda else x

def toCpu(x):
	if type(x) is tuple:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()

n_cuda_devices = torch.cuda.device_count()

# First order derivatives (d/dx)

dx_kernel_copies = [toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2), cuda_i) for cuda_i in range(n_cuda_devices)]
def dx(v):
	return F.conv2d(v,dx_kernel_copies[current_cuda],padding=(0,1)) / params.space_unit

dx_left_kernel_copies = [toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2), cuda_i) for cuda_i in range(n_cuda_devices)]
def dx_left(v):
	return F.conv2d(v,dx_left_kernel_copies[current_cuda],padding=(0,1)) / params.space_unit

dx_right_kernel_copies = [toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2), cuda_i) for cuda_i in range(n_cuda_devices)]
def dx_right(v):
	return F.conv2d(v,dx_right_kernel_copies[current_cuda],padding=(0,1)) / params.space_unit

# First order derivatives (d/dy)

dy_kernel_copies = [toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3), cuda_i) for cuda_i in range(n_cuda_devices)]
def dy(v):
	return F.conv2d(v,dy_kernel_copies[current_cuda],padding=(1,0)) / params.space_unit

dy_top_kernel_copies = [toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3), cuda_i) for cuda_i in range(n_cuda_devices)]
def dy_top(v):
	return F.conv2d(v,dy_top_kernel_copies[current_cuda],padding=(1,0)) / params.space_unit

dy_bottom_kernel_copies = [toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3), cuda_i) for cuda_i in range(n_cuda_devices)]
def dy_bottom(v):
	return F.conv2d(v,dy_bottom_kernel_copies[current_cuda],padding=(1,0)) / params.space_unit

# Curl operator

### IMPORTANT: IF SPACE_UNIT != 1, THEN ALL a's TRUE VALUES WILL BE SCALED BY 1 / params.space_unit
def rot_mac(a):
	return torch.cat([-dx_right(a) * params.space_unit,dy_bottom(a) * params.space_unit],dim=1)

# Laplace operator

#laplace_kernel = toCuda(torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]]).unsqueeze(0).unsqueeze(1)) # 5 point stencil
#laplace_kernel = toCuda(torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]]).unsqueeze(0).unsqueeze(1)) # 9 point stencil
laplace_kernel_copies = [toCuda(0.25*torch.Tensor([[1,2,1],[2,-12,2],[1,2,1]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)] # isotropic 9 point stencil
def laplace(v):
	return F.conv2d(v,laplace_kernel_copies[current_cuda],padding=(1,1)) / (params.space_unit**2)


# mapping operators

map_vx2vy_kernel_copies = [0.25*toCuda(torch.Tensor([[0,1,1],[0,1,1],[0,0,0]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)]
def map_vx2vy(v):
	return F.conv2d(v,map_vx2vy_kernel_copies[current_cuda],padding=(1,1))

map_vx2vy_left_kernel_copies = [0.5*toCuda(torch.Tensor([[0,1,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)]
def map_vx2vy_left(v):
	return F.conv2d(v,map_vx2vy_left_kernel_copies[current_cuda],padding=(1,1))

map_vx2vy_right_kernel_copies = [0.5*toCuda(torch.Tensor([[0,0,1],[0,0,1],[0,0,0]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)]
def map_vx2vy_right(v):
	return F.conv2d(v,map_vx2vy_right_kernel_copies[current_cuda],padding=(1,1))

map_vy2vx_kernel_copies = [0.25*toCuda(torch.Tensor([[0,0,0],[1,1,0],[1,1,0]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)]
def map_vy2vx(v):
	return F.conv2d(v,map_vy2vx_kernel_copies[current_cuda],padding=(1,1))

map_vy2vx_top_kernel_copies = [0.5*toCuda(torch.Tensor([[0,0,0],[1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)]
def map_vy2vx_top(v):
	return F.conv2d(v,map_vy2vx_top_kernel_copies[current_cuda],padding=(1,1))

map_vy2vx_bottom_kernel_copies = [0.5*toCuda(torch.Tensor([[0,0,0],[0,0,0],[1,1,0]]).unsqueeze(0).unsqueeze(1), cuda_i) for cuda_i in range(n_cuda_devices)]
def map_vy2vx_bottom(v):
	return F.conv2d(v,map_vy2vx_bottom_kernel_copies[current_cuda],padding=(1,1))


mean_left_kernel_copies = [0.5*toCuda(torch.Tensor([1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2), cuda_i) for cuda_i in range(n_cuda_devices)]
def mean_left(v):
	return F.conv2d(v,mean_left_kernel_copies[current_cuda],padding=(0,1))

mean_top_kernel_copies = [0.5*toCuda(torch.Tensor([1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3), cuda_i) for cuda_i in range(n_cuda_devices)]
def mean_top(v):
	return F.conv2d(v,mean_top_kernel_copies[current_cuda],padding=(1,0))

mean_right_kernel_copies = [0.5*toCuda(torch.Tensor([0,1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2), cuda_i) for cuda_i in range(n_cuda_devices)]
def mean_right(v):
	return F.conv2d(v,mean_right_kernel_copies[current_cuda],padding=(0,1))

mean_bottom_kernel_copies = [0.5*toCuda(torch.Tensor([0,1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3), cuda_i) for cuda_i in range(n_cuda_devices)]
def mean_bottom(v):
	return F.conv2d(v,mean_bottom_kernel_copies[current_cuda],padding=(1,0))


def staggered2normal(v):
	ret = torch.clone(v)
	ret[:,0:1] = mean_left(v[:,0:1])
	ret[:,1:2] = mean_top(v[:,1:2])
	return ret

def normal2staggered(v):#CODO: double-check that! -> seems correct
	ret = torch.clone(v)
	ret[:,0:1] = mean_right(v[:,0:1])
	ret[:,1:2] = mean_bottom(v[:,1:2])
	return ret



def vector2HSV(vector,plot_sqrt=False):
	"""
	transform vector field into hsv color wheel
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = torch.ones(values.shape).cuda(current_cuda)
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()
