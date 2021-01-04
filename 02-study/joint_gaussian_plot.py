import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

rstate = 1337

# set fonts
hfont = {'fontname':'Helvetica', 'size': 20}
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

min = -5
max = 5

# Our 2-dimensional distribution will be over variables X and Y
N = 60
x = np.linspace(min, max, N)
X = np.linspace(min, max, N)
y = np.linspace(min, max, N)
Y = np.linspace(min, max, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 1.5 , -0.5], [-0.5,  1.25]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y



# The distribution on the variables X, Y packed into pos.
F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure(figsize=(18, 10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0.1, antialiased=True, cmap=cm.viridis)


# plot marginals
marg_y = norm.pdf(y, mu[1],Sigma[1,1])
ax.plot(y, marg_y, zs=5, zdir='y', label=r'$p(x_2)$')
#ax.text(-5, -0.75, 0.3, zdir='y', s=r'$p(x_2)$', fontsize=18)
ax.text2D(0.285, 0.720, r'$p(x_2)$', transform=ax.transAxes, fontsize=20)

marg_x = norm.pdf(x, mu[0],Sigma[0,0])
ax.plot(x, marg_x, zs=-5, zdir='x', label=r'$p(x_1)$')
#ax.text(-0.4, 5, 0.325, zdir='x', s=r'$p(x_1)$', fontsize=18)
ax.text2D(0.620, 0.815, r'$p(x_1)$', transform=ax.transAxes, fontsize=20)



# Adjust the limits, ticks and view angle
ax.set_zlim(0, 0.35)
ax.set_zticks(np.linspace(0, 0.35, 5))

ax.set_xlabel(r'$x_1$', labelpad=10, **hfont)
ax.tick_params(axis='x', which='both', labelsize=20)

ax.set_ylabel(r'$x_2$', labelpad=10,**hfont)
ax.tick_params(axis='y', which='both', labelsize=20)

ax.set_zlabel(r'$p$',labelpad=30, **hfont)
ax.tick_params(axis='z', which='both', labelsize=20, pad=15)


ax.view_init(20, -60)
plt.tight_layout()
plt.savefig('joint_gaussian.png', dpi=300)
plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.20, cmap=cm.viridis, antialiased=True)
# ax.set_xlabel(r'$x_1$', labelpad=10, **hfont)
# ax.tick_params(axis='x', which='both', labelsize=20)
#
# ax.set_ylabel(r'$x_2$', labelpad=10,**hfont)
# ax.tick_params(axis='y', which='both', labelsize=20)
# cbar = plt.colorbar(cset, shrink=0.75)
# cbar.ax.tick_params(labelsize=18)
# cbar.set_label(r'$p(x_1, x_2)$', fontsize=20, labelpad=10)
# plt.tight_layout()
# plt.savefig('contour_gaussian.png', dpi=300)

plt.show()
