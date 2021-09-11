# Real-World-Softbody

**By Paper ID : 2591**


### Introduction
This repository accompanies the paper "Flexible Networks for Learning Physical Dynamics of Deformable Objects", submitted to the main track of "AAAI 2022" with paper id 2591.<br/>
We release the code to train, test, and visualize the result of our model.<br/>
Note that the data attached in this repository is a small subset of the original dataset we used for training, due to file size limit.<br/>
The implementation is based on python 3.6, tensorflow 2.3.0., CUDA 10.1, and cuDNN 7.6.1 <br/>

### How To Run
**1. Configure Environment**


    pip install -r requirements.txt

**2. Preparing Synthetic Dataset**
<br/>Due to file size limit, the data attatched in this repo is a small subset of the original data. <br/>
You can either use this small subset of the synthetic dataset (go to 2-1), <br/>
or generate and preprocess the entire synthetic dataset from scratch to reproduce the result with best performance indicated in our paper (go to 2-2). <br/>
The entire process of generating the synthetic dataset takes a couple of hours and consumes approximately 12.43GB.

**2-1. Using small dataset**

    # unzip synthetic_dataset_1.zip, synthetic_dataset_2.zip and place it under the root directory of this repo

    mv ./synthetic_dataset_1 ./data/synthetic_dataset
    mv ./synthetic_dataset_2/y_train_pred_ordered.json ./data/synthetic_dataset/input_4/offset_4_num_pred_8/y_train_pred_ordered.json

**2-2. Generating the entire Synthetic Dataset**

    python3 box2d_simulator/simulator.py                     # generates raw point set data
    python3 data/simulation/preprocess_code/preprocess.py    # preprocess data

**3. Preparing Real-World Dataset**

    # unzip real_world_dataset.zip and place it under the root directory of this repo
    mv ./real_world_dataset ./data/real_world_dataset

**4. Train**
<br/>To train TP-Net with the parameters that we used for getting the best performance, execute the following command.
<br/>You can change the hyperparameters or other training options by changing config.py.


    CUDA_VISIBLE_DEVICES=0 python3 train.py

**5. Evaluate & Visualize**
<br/>To evaluate the trained model on test cases, run 

    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation/evaluate_synthetic.py --init_data_type=ordered
    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation/evaluate_real_world.py --init_data_type=unordered

<br/>To visualize the results, run
    
    python3 ./evaluation/visualize_synthetic.py
    python3 ./evaluation/visualize_real_world.py
