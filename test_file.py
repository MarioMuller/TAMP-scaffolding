import robotic as ry

C = ry.Config()
f = C.addFrame("test")
f.setPosition([0.5, 0, 0.5])
f.setShape(ry.ST.cylinder, [1, 0.03])   # maybe [radius, length]
C.view()
input("Press Enter to close...")