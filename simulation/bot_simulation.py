from pygame import *
from simulation.rendering import camera, shape, light_source
import numpy as np

screen = display.set_mode((800, 600))

c = camera.Camera(800, 600, 90)

cyl = shape.cylinder.Cylinder(10, 1, faces=50, pos=np.array((20, -0.5, 0, 1)))

#cyl.rotate_x(np.radians(90))
#cyl.translate_x(20)

running = True

l = light_source.LightSource(np.array((5, 5, 5, 1)))

while running:
    for e in event.get():
        if e.type == QUIT:
            running = False
    pols = c.render_shapes([cyl], l)
    pols.sort(key=lambda x: sum(x.face[:, 0])/4)
    screen.fill((80, 90, 100))
    for pol in pols:

        draw.polygon(screen, pol.colour, pol.face[:, 1:3])
    cyl.rotate_y(np.radians(0.2))

    display.flip()

quit()