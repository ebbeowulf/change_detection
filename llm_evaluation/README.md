----------------- How to install ---------------------
Create a conda environment

conda create -n llama10 python=3.10
conda activate llama10

Install the python package:
pip install -e .

or to update:

pip install -e . --force-reinstall --no-deps

----------------- How to use ---------------------
This work is NOT going to work with the other libraries in ROS. It needs CUDA 12.2 and is not compatible
with huggingface. So I've set it up as a standalone repo that is communicated with by sockets. 

Server:
python llm_server.py # By default uses localhost and port 5000

Client Example Code:
python llm_client.py

The server will accept images and text. You can also change which model is being used. The idea is to expand 
the repo in the future as we integrate with more models, but preserve the general client/server architecture.
