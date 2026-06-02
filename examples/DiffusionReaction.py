import openpnm as op                                                  # noqa: E402
import scipy, scipy.linalg, scipy.sparse                              # noqa: E401, E402
import pnm_ice.models.const_spheres_and_cylinders as geo_model    # noqa: E402
import numpy as np                                                    # noqa: E402
from pnm_ice.ToolSet import MulticomponentTools                   # noqa: E402
import pnm_ice.Operators as ops                                   # noqa: E402
import pnm_ice.Interpolation as ip                                # noqa: E402
import pnm_ice.BoundaryConditions as bc                           # noqa: E402
from pnm_ice.NumericalDifferentiation import conduct_numerical_differentiation
from matplotlib import pyplot as plt                                   # noqa: E402


Nx = 20
Ny = 5
Nz = 1
Nc = 2
spacing = 1./Nx

# get network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

# add geometry
geo = geo_model.spheres_and_cylinders
network.add_model_collection(geo, domain='all')
network.regenerate_models()

c = np.zeros((network.Np, Nc))
D_0 = 1e-5
D_1 = 5e-5

mt = MulticomponentTools(network=network, num_components=Nc)
bc.set(mt, label='left', id=0, bc=1.)
bc.set(mt, label='left', id=1, bc=0.)
bc.set(mt, label='right', id=0, bc=0.)
bc.set(mt, label='right', id=1, bc=1.)

x = np.ndarray.flatten(c).reshape(-1, 1)

pos = 0
tol = 1e-6
max_iter = 100

def defect_reac(V, c):
    k = 1
    c_corr = np.copy(c)
    c_corr[c_corr < 0] = c_corr[c_corr < 0] **2
    r = k * c_corr[:, 0]**2 * c_corr[:, 1] * V.reshape(-1)
    G = np.zeros_like(c)
    G[:, 0] = -r
    G[:, 1] = -r
    return G

grad = ops.gradient(mt)
sum = ops.sum(mt)

D = np.zeros((network.Nt, Nc), dtype=float)
D[:, 0] = D_0
D[:, 1] = D_1
J_diff = sum(D, grad)

for i in range(max_iter):
    pos += 1
    J_r, G_r = conduct_numerical_differentiation(c=x.reshape(-1, Nc),
                                                 defect_func = lambda x: defect_reac(mt.get_network()['pore.volume'], x),
                                                 type='constrained')
    J = J_diff + J_r
    J = bc.apply(mt, A=J)
    G = J_diff * x + G_r
    G = bc.apply(mt, x=x, b=G, type='Defect')
    last_iter = i
    G_norm = np.linalg.norm(np.abs(G), ord=2)
    print(f'{last_iter + 1} it - G: {G_norm}')
    if G_norm < tol:
        break
    dx = scipy.sparse.linalg.spsolve(J, -G).reshape(-1, 1)
    x = x + dx
    x[x < 0] = 0

if last_iter == max_iter - 1:
    print(f'WARNING: the maximum iterations ({max_iter}) were reached!')


c_0 = x.reshape(-1, Nc)[:, 0]
c_1 = x.reshape(-1, Nc)[:, 1]

plt.plot(mt.get_network()['pore.coords'][:, 0], c_0, 'o', label='c_0')
plt.plot(mt.get_network()['pore.coords'][:, 0], c_1, 'x', label='c_1')
plt.legend()
plt.show()

print('finished')
