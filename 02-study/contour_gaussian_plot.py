import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

rstate = 1337

# set fonts
hfont = {'fontname':'Helvetica', 'size': 18}
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
fig, ax = plt.subplots()

#plt.figure(figsize=(14,10))

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.20, cmap=cm.viridis, antialiased=True)
#ax.clabel(CS, inline=1, fontsize=10)
#ax.set_title('Simplest default with labels')
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0.1, antialiased=True, cmap=cm.viridis)



# plot marginals
# marg_y = norm.pdf(y, mu[1],Sigma[1,1])
# ax.plot(y, marg_y, zs=5, zdir='y', label=r'$p(x_2)$')
# #ax.text(-5, -0.75, 0.3, zdir='y', s=r'$p(x_2)$', fontsize=18)
# ax.text2D(0.280, 0.725, r'$p(x_2)$', transform=ax.transAxes, fontsize=18)
#
# marg_x = norm.pdf(x, mu[0],Sigma[0,0])
# ax.plot(x, marg_x, zs=-5, zdir='x', label=r'$p(x_1)$')
# #ax.text(-0.4, 5, 0.325, zdir='x', s=r'$p(x_1)$', fontsize=18)
# ax.text2D(0.615, 0.815, r'$p(x_1)$', transform=ax.transAxes, fontsize=18)



# Adjust the limits, ticks and view angle

#plt.legend(fontsize=18)
#cbar = plt.colorbar(cset, label=r'$p(x_1, x_2)$', shrink=0.5)
#cbar.ax.tick_params(labelsize=18)
plt.show()
#plt.savefig('gaussian_plot.png', dpi=600)

