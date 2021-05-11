# Real-World-Softbody

**By <a href="http://github.com/jinhyung426/" target="_blank">Jinhyung Park</a>, Dohae Lee from Yonsei University (Seoul, Republic of Korea)**

<p align="center">
  <img width="912" height="441" src="https://github.com/cgna-soft/Real-World-Softbody/blob/main/utils/teaser.jpg">
</p>
<br/>


### Introduction
- This repository accompanies the paper "Paper Title TBD", submitted to "Journal Name TBD".<br/>
  We release the code and data to train, test, and visualize the result of our model.<br/>
  The implementation is based on python 3.8 and tensorflow 2.3.0. <br/>

### How To Run
**1. Configure Environment**


    pip install -r requirements.txt
   
**2. Download Dataset**
    
    # command for downloading dataset

**3. Train**


    python3 train_pred.py

**4. Test**


    python3 test_pred.py

### Evaluation
- Our model is compared to the following models that are capable of predicting future frames of real world softbody.<br/>

   -  <a href="https://github.com/cgraber/cvpr_dNRI" target="_blank">Dynamic Neural Relational Inference (CVPR, 2020)</a>   
   -  <a href="https://github.com/seuqaj114/paig" target="_blank">Physics As Inverse Graphics (ICLR, 2020)</a>
   -  <a href="https://github.com/mbchang/dynamics" target="_blank">Neural Physics Engine (ICLR, 2017)</a>
   -  <a href="https://github.com/jinhyung426/Interaction-networks_tensorflow" target="_blank">Interaction Network (NIPS, 2016)</a>
