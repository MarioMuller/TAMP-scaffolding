import robotic as ry
import numpy as np
import time
print('robotic package version:', ry.compiled())

C = ry.Config()
C.clear()
# C.addFile("/home/mario/TAMP-scaffolding/src/models/ur10/ur10.g")
# C.addFile("/home/mario/TAMP-scaffolding/src/models/ur5/ur5.g")
# C.addFile("/home/mario/TAMP-scaffolding/src/models/panda/panda.g")
C.addFile("/home/mario/TAMP-scaffolding/src/models/husky/husky.g")
# C.addFile("/home/mario/multirobot-pathplanning-benchmark/src/multi_robot_multi_goal_planning/assets/models/rai/husky/husky.g")
# C.addFile(ry.raiPath('ur5/ur5.g'))
C.view()
time.sleep(50)