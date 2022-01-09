# Flexible Networks for Learning Physical Dynamics of Deformable Objects

**By <a href="http://github.com/jinhyung426/" target="_blank">Jinhyung Park</a>, <a href="https://github.com/dlehgo14" target="_blank">Dohae Lee</a>, In-Kwon Lee from Yonsei University (Seoul, Korea)**<br/>

To download our paper, click <a href="https://arxiv.org/pdf/2112.03728" target="_blank">here</a>. <br/>

### Introduction

<p align="center">
  <img width="720" height="250" src="https://github.com/jinhyung-park-info/TP-Net/blob/main/utils/main_teaser.png">
</p>

This repository accompanies the paper "Flexible Networks for Learning Physical Dynamics of Deformable Objects", which is currenty under review for publication.<br/>
We release the code to train, test, and visualize the result of our model.<br/>
The implementation is based on python 3.6, tensorflow 2.3.0., CUDA 10.1, and cuDNN 7.6.1 <br/>

### How To Run
**1. Configure Environment**

    pip install -r requirements.txt
<br/>

**2-1. Download Dataset**
<br/> Download each dataset from the links below. <br/>
- Synthetic Dataset: https://drive.google.com/file/d/1VcwW4UVLUFseo1iJfCdswjAgnFU87ETg/view?usp=sharing <br/> 
- Real-World Dataset: https://drive.google.com/file/d/1svoipJCZ6Amz04-aKn7rwWWUPEmeQop-/view?usp=sharing <br/>

After downloading and unzipping each dataset, place each folder as below. <br/>

    data/synthetic_dataset/preprocessed_data
    data/real_world_dataset/preprocessed_data
<br/>  

**2-2 (Alternative) Generating the entire Synthetic Dataset** <br/>
Alternatively, you can generate the synthetic dataset from scratch by executing the following commands. <br/>
The entire process of generating the synthetic dataset takes a couple of hours and consumes approximately 12.43GB.

    python3 box2d_simulator/simulator.py                     # generates raw point set data
    python3 data/simulation/preprocess_code/preprocess.py    # preprocess data
<br/>

**3. Train**
<br/>To train TP-Net with the parameters that we used for getting the best performance, execute the following command.
<br/>You can change the hyperparameters or other training options by changing config.py.


    CUDA_VISIBLE_DEVICES=0 python3 train.py
<br/>

**4. Evaluate & Visualize**
<br/>To evaluate the trained model on test cases, run 

    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation/evaluate_synthetic.py --init_data_type=ordered
    CUDA_VISIBLE_DEVICES=0 python3 ./evaluation/evaluate_real_world.py --init_data_type=unordered

To visualize the results, run <br/>

    python3 ./evaluation/visualize_synthetic.py
    python3 ./evaluation/visualize_real_world.py
