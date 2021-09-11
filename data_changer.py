import os
import shutil

path = os.path.join('..', 'post_processed_data')

for case in os.listdir(path):
    basepath1 = os.path.join(path, case, 'contours')
    basepath2 = os.path.join(path, case, 'points')
    if os.path.isdir(basepath1):
        shutil.rmtree(basepath1)
        shutil.rmtree(basepath2)
