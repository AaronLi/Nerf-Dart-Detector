import numpy as np
import matplotlib.pyplot as plt


def get_polar(c1, c2, ori=np.zeros(2)):
    m = (c1[1]-c2[1])/(c1[0]-c2[0]) #finds slope of line between two points
    b = c1[1]-m*c1[0] # finds y-intercept of line between points
    m_orth = -1/m # finds slope of intercept
    x = b/(m_orth-m) # finds POI
    y = m_orth*x
    
    r = np.hypot(x,y) # calculates distance from origin
    theta = np.arctan2(y,x) # calculates angle from positive x-axis
    return r,theta 


if __name__ == '__main__':  
    polar = []
    c1 = np.array([0,4])
    c2 = np.array([3,0])
    r,theta = get_polar(c1, c2)
    print(r)
    print(theta)

    xs1 = [c1[0],c2[0]]
    ys1 = [c1[1],c2[1]]
    xs2 = [0,r*np.cos(theta)]
    ys2 = [0,r*np.sin(theta)]
    plt.xlim(-0.5,5)
    plt.ylim(-0.5,5)
    plt.axis('equal')
    plt.plot(xs1,ys1,"rd-")
    plt.plot(xs2,ys2,"bo-")
    plt.show()

