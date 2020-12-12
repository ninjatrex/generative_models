import os

output_path = '../lib/images/plots'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# implementing a funtion for creating a single normal distribution
import math

SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    return math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma)

# lets plot some pdfs
import matplotlib.pyplot as plt
import numpy as np
# points xs ranging from -5 to 5
xs = [x / 10.0 for x in range(-100, 100)]
# actual plot
plt.figure(figsize=(10, 5))

first = [normal_pdf(x,sigma=0.75, mu=-4.5) for x in xs]
second = [normal_pdf(x,sigma=2, mu=0) for x in xs]
third = [normal_pdf(x,sigma=1.25, mu=4) for x in xs]

weights = np.array([16, 4, 12])

combination = np.array([first, second, third])

gmm_naive = np.sum(combination, axis=0)

plt.plot(xs, gmm_naive, '-', label='GMM density', color='k')

plt.plot(xs, first,'--',label='mu=-4,5,sigma=0.75', color='tab:blue')
plt.plot(xs, second,'--',label='mu=0,sigma=2', color='tab:orange')
plt.plot(xs, third,'--',label='mu=4,sigma=1.25', color='tab:green')
plt.legend()
plt.ylabel('p(x)')
plt.xlabel('x')
plt.savefig('{}/gmm.png'.format(output_path), dpi=300)
