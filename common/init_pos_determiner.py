import random
import matplotlib.pyplot as plt
from common.Constants import INIT_POS_MAX, INIT_POS_MIN, POS_LST
SEED = 2
random.seed(SEED)
TEST_POS_LST = []

# Normal
while len(TEST_POS_LST) < 75:
    x = random.randrange(280, int(INIT_POS_MAX * 10)) / 10
    y = random.randrange(280, int(INIT_POS_MAX * 10)) / 10

    assert INIT_POS_MIN <= x <= INIT_POS_MAX
    assert INIT_POS_MIN <= y <= INIT_POS_MAX
    if (x, y) not in TEST_POS_LST and (x, y) not in POS_LST:
        TEST_POS_LST.append((x, y))

TEST_POS_LST.sort()
print(TEST_POS_LST)

xs = []
ys = []

for init_x_pos, init_y_pos in TEST_POS_LST:
    xs.append(init_x_pos)
    ys.append(init_y_pos)

plt.scatter(xs, ys)
plt.axis([0, 42.5, 0, 42.5])
plt.savefig('Initial_Test_Positions.png')

f = open('../utils/Initial_Test_Positions.txt', 'w')
f.write(f'Date : 2021_06_22_00:16\n')
f.write(f'Random Seed : {SEED}\n')

for i in range(len(TEST_POS_LST)):
    f.write(f'{TEST_POS_LST[i]}\n')
f.close()
