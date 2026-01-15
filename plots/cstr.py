import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Function and derivative
def f(C):
    return 0.5*C**2 - C + 1

def df(C):
    return C - 1

# Safe Newton-Raphson iteration
C0 = 0.0            # Initial guess (avoid df=0)
iterations = 10
tol = 1e-6
C_vals = [C0]

for i in range(iterations):
    derivative = df(C0)
    if derivative == 0:
        print(f"Derivative zero at iteration {i}, C = {C0}. Stopping iteration.")
        break
    C1 = C0 - f(C0)/derivative
    C_vals.append(C1)
    if abs(C1 - C0) < tol:
        break
    C0 = C1

# Animation plot
C_range = np.linspace(-1, 2, 400)
F_vals = f(C_range)

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(C_range, F_vals, 'k', linewidth=2, label='f(C)')
ax.axhline(0, color='gray', linestyle='--')
point, = ax.plot([], [], 'ro', markersize=8)
line, = ax.plot([], [], 'r--', alpha=0.5)
ax.set_xlabel('C')
ax.set_ylabel('f(C)')
ax.set_title('Newton-Raphson Iteration')
ax.legend()
ax.grid(True)

def update(frame):
    xdata = [C_vals[frame], C_vals[frame]]
    ydata = [0, f(C_vals[frame])]
    line.set_data(xdata, ydata)
    point.set_data(C_vals[frame], f(C_vals[frame]))
    ax.set_title(f"Iteration {frame}")
    return point, line

anim = FuncAnimation(fig, update, frames=len(C_vals), interval=700, blit=True)
anim.save("chapter4_newton_raphson_safe.gif", writer=PillowWriter(fps=1))
plt.show()
