import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data = np.genfromtxt("data/2d.csv", delimiter=",")
N = len(data)



def least_squar_err(X, Y, theta):
  return X.T.dot(X.dot(theta) - Y) / N

  # theta_0 = theta[0]
  # theta_1 = theta[1]
  # t_0 = 0
  # t_1 = 0
  # for i in range(0, N):
  #   x = data[i][0]
  #   y = data[i][1]
  #   t_0 += theta_0 + theta_1 * x - y
  #   t_1 += (theta_0 + theta_1 * x - y) * x
  # t_0 = 1.0 / N * t_0
  # t_1 = 1.0 / N * t_1
  # return np.array([t_0, t_1])

def gradient_descent_runner(learning_rate, num_iterations, cost_func, X, Y, theta):
    for i in range(num_iterations):
      theta = theta - learning_rate * cost_func(X, Y, theta)
    return theta

def create_line(intercept, slope):
  x = range(0,100)
  y = [slope * i + intercept for i in x]
  return [x,y]

g_theta = np.array([[1], [1]])

# print gradient_descent_runner(0.0001, 1000, least_squar_err, np.array([[1, i] for i in data[:,0]]), data[:,1].reshape(N, 1), g_theta)

if __name__ == '__main__':

  fig, ax = plt.subplots()
  ax.scatter(data[:,0], data[:,1])

  x , y = create_line(0,0)
  line, = ax.plot(x, y)



  def animate(i):
    global g_theta
    g_theta = gradient_descent_runner(0.0001, 1, least_squar_err, np.array([[1, i] for i in data[:,0]]), data[:,1].reshape(N, 1), g_theta)
    print g_theta
    x , y = create_line(g_theta[0], g_theta[1])
    line.set_ydata(y)
    return line,

  def init():
    x , y = create_line(0,0)
    line.set_ydata(y)
    return line,

  ani = animation.FuncAnimation(fig, animate, np.arange(1, 20), init_func=init,
                                interval=50, repeat=False, blit=False)

  # Closed form
  X = np.array([[1, i] for i in data[:,0]])
  Y = data[:,1].reshape(N, 1)
  g_theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)
  print("closed form", g_theta)
  x , y = create_line(g_theta[0], g_theta[1])
  ax.plot(x, y, "r")
  
  g_theta = np.array([[0], [0]])
  plt.show()

  


