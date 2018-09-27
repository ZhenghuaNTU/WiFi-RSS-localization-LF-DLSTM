# WiFi-RSS-localization-LF-DLSTM

The file 'LF_extraction.m' is for local feature extraction. 

The file 'getSimData.m' generates the data for simulation based on the layout of the research lab. It includes the data generation and local feature extraction, and fianlly save the data as 'SimData1.mat' for the running of the DLSTM. 

The file 'LF-DLSTM.py' is the DLSTM.

The running includes two steps:
Firstly, run 'getSimData.m' with Matlab to get the data (after local feature extraction). 
Then, run 'LF-DLSTM.py'for localization.

Requirement: 

Matlab 2015b   
Python 3.6  
Tensorflow 1.4.1   
Keras 2.1.4
