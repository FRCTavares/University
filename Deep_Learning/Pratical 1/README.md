# Welcome to the Deep Learning Course (2024-2025) Praticals Instructions for Running Notebooks

Below you can find 
- [how to install the conda environment and all required packages to run the practical sessions on your computer](##installation-of-virtual-environment-for-pratical-sessions)
- [how to use Google Colab to run the practical sessions](##)

## Installation of virtual environment for Pratical Sessions
#### 1. To set the virtual environment to run these pratical sessions we suggest the use conda. If you do not have you can install one of these two options:
  - [Anaconda (management system with some packages already built in)](https://docs.anaconda.com/anaconda/install/)
  - [Miniconda (management system without packages)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

#### 2. Download the environment.yml file and use it to create the virtual environment: 
  - download the environment.yml to a directory of your choice
  - \[Windows\] open command line and change the current working directory to the one containing the environment file (example below for the case where the file was downloaded to the Downloads directory - replace <current_user> by your user name):
  ```
    cd C:\Users\<current_user>\Downloads\
  ```
  - \[Mac/Linux\] open command line and change the current working directory to the one containing the environment file (example below for the case where the file was downloaded to the Downloads directory - replace <current_user> by your user name):
  ```
    cd ~/Downloads
  ```
  - create environment (called IST_DL21_Env) and install required packages automatically:
  ```
    conda env create -f environment.yml
  ```

### Activate environment and start jupyter notebook:
  ```
    conda activate IST_DL21_Env
    jupyter notebook
  ```

## Use Google Colaboratory to run the Pratical Sessions (requires Google Account)

Use the following link https://colab.research.google.com/ to open Google Colaboratory (Colab)

One of the two following windows will be visible to you:

1. 
![image](https://drive.google.com/uc?export=view&id=122-tpn4ed2BultopLdz_FawBzskCCL1j)

2.
![image](https://drive.google.com/uc?export=view&id=1pcVmxji53WDThx5cIHFdorlOWBvDRs7E)

In scenario 1. you can open a new notebook by clicking on the button highlighted in the figure below

![image](https://drive.google.com/uc?export=view&id=1hqPFyUBDqxXzky7Ly0cjsRqnIYQeYARW)

In scenario 2. you can open a new notebook by going to File -> New notebook as shown in the figure below

![image](https://drive.google.com/uc?export=view&id=1e3gLgyaivmadWUg7lYJ4PJI9N0C0bQbB)

### You can also open a notebook saved in your Google Drive or upload an existing jupyter notebook by going to File -> Open notebook/Upload notebook which will bring you to the following two windows (the open to open Recent notebooks, notebooks present in your Google Drive, and notebooks available in Github repositories).

![image](https://drive.google.com/uc?export=view&id=13f73tuf-sN4Yvxmt7g6x06tWxPWb2nWW)
![image](https://drive.google.com/uc?export=view&id=1qtRsStdt385TC1IH0Nh39ku_J67n-KkP)

### Colab has CPU, GPU and TPU runtime types available for you to run your notebooks. While on the first praticals you may only need a CPU instance, from practical 5 you may take advantage of a GPU instance. To change runtime type you can do as follows.

![image](https://drive.google.com/uc?export=view&id=1VY3Db62VAVLsGae5CpasG9uUsJxqa_R3)
![image](https://drive.google.com/uc?export=view&id=1PXhZnaAwCrdi5ZpTMqWia87xpg2ImuS1)
