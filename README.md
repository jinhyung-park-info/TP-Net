# Real-World-Softbody

**By Paper ID : 2591**


### Introduction
This repository accompanies the paper "Flexible Networks for Learning Physical Dynamics of Deformable Objects", submitted to the main track of "AAAI 2022".<br/>
We release the code and data to train, test, and visualize the result of our model.<br/>
The implementation is based on python 3.6, tensorflow 2.3.0., CUDA 10.1, and cuDNN  <br/>

### How To Run
**1. Configure Environment**


    pip install -r requirements.txt
   
**2-1. Download Dataset**
Due to file size limit, please download our synthetic dataset from the following link.
URL : 

Assuming the data is downloaded in the root directory of this repo, move the data files using the command below.
    mv ./raw_data ./data/simulation/raw_data/pointset
    mv ./preprocessed_data ./data/simulation/preprocessed_data

**2-2. Generating Synthetic Dataset from Scratch**
If you want to generate and preprocess the entire synthetic dataset from scratch, execute the following commands. The entire process takes a couple of hours.


    python3 box2d_simulator/simulator.py                     # generates raw data
    python3 data/simulation/preprocess_code/preprocess.py    # preprocess data

**3. Train**
To train TP-Net with default parameters, execute the following command.
You can change the hyperparameters or other training options by changing config.py.

    CUDA_VISIBLE_DEVICES=0 python3 train.py

**4. Evaluate & Visualize**
To evaluate the trained model on test cases, run 

    python3 ./evaluation/evaluate.py

To visualize the rollout, run
    
    python3 ./evaluation/visualize_synthetic.py
    python3 ./evaluation/visualize_real_world.py

