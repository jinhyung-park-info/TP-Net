import random
import matplotlib.pyplot as plt
from common.Constants import *
SEED = 2
random.seed(SEED)
POS_LST = []

# Normal
while len(POS_LST) < 75:
    x = random.randrange(280, int(INIT_POS_MAX * 10)) / 10
    y = random.randrange(280, int(INIT_POS_MAX * 10)) / 10

    assert INIT_POS_MIN <= x <= INIT_POS_MAX
    assert INIT_POS_MIN <= y <= INIT_POS_MAX
    if (x, y) not in POS_LST:
        POS_LST.append((x, y))

POS_LST.sort()
print(POS_LST)

xs = []
ys = []

for init_x_pos, init_y_pos in POS_LST:
    xs.append(init_x_pos)
    ys.append(init_y_pos)

plt.scatter(xs, ys)
plt.axis([0, 42.5, 0, 42.5])
plt.savefig('Initial_Positions.png')

f = open('../utils/Initial_Positions.txt', 'w')
f.write(f'Date : 2021_04_40_03:54\n')
f.write(f'Random Seed : {SEED}\n')

for i in range(len(POS_LST)):
    f.write(f'{POS_LST[i]}\n')
f.close()
