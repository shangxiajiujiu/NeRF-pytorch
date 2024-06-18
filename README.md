# NeRF-pytorch
This is a coursework assignment，a project with a NeRF implementation.

![image](https://github.com/shangxiajiujiu/NeRF-pytorch/assets/72213981/c67c48f4-a3dd-430a-b2bc-644a40842d31)

# Videos
This is a demo video of the project introduction and deployment  

https://github.com/shangxiajiujiu/NeRF-pytorch/assets/72213981/43e69ead-b6a7-4bf0-94ec-a343d9ff5b5c



# Dependencies
PyTorch 1.4  
matplotlib  
numpy  
imageio  
imageio-ffmpeg  
configargparse  
Please execute the following command to download the package  
“pip install -r requirements.txt”  

# Datasets  
Please use the following command to execute the bash file to download the dataset  
“bash download_example_data.sh”  
Or you can go directly to this website to download the dataset  
“https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/”  
Once the download is complete, unzip the folder and place it in the “data” folder.  

# Training  
Please use the following command to run  
“python run_nerf.py --config configs/xxx.txt”  
Please open the configs/txt file to see the detailed parameter settings.  

After training for 200k iterations (7~8 hours on a single 40 Series GPU), you can find the following video at `logs/xxx/xxx.mp4` and the following images at `logs/xxx/xxx.png`
