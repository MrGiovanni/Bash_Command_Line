# Connect to ASU VPN
##### 1. Open Cisco AnyConnect Secure Mobility Client
Type in: `sslvpn.asu.edu` and click `Connect`

##### 2. Type in user name (asu email) and password

##### 3. Remotely access the lab machines
```bash
ssh zongwei@r3-host.hfc.dhcp.asu.edu -X
exit
scp -r zongwei@r3-host.hfc.dhcp.asu.edu:/mnt/dfs/zongwei /Users/zongwei.zhou/
scp zongwei@r3-host.hfc.dhcp.asu.edu:/mnt/dfs/zongwei/debug.py /Users/zongwei.zhou/
```

# Access ASU GPU cluster
- Credit to [ASU Research Computing](https://rcstatus.asu.edu/agave/)
- For any assistance, please contact: [support@hpchelp.asu.edu](support@hpchelp.asu.edu)
- Please check [Requesting Resources on Sol](https://asurc.atlassian.net/wiki/spaces/RC/pages/1642102826/Requesting+Resources+on+Sol)
##### Create an account at https://cores.research.asu.edu/research-computing/about-rc then you are good to go
```bash
ssh -Y zzhou82@agave.asu.edu
ssh -Y zzhou82@login.sol.rc.asu.edu
scp /Users/zongwei.zhou/debug.py zzhou82@agave.asu.edu:/home/zzhou82/zongwei.zhou/ # Transferring files
pip install simpleitk photutils tifffile libtiff pydot --user
sbatch --error=logs/run_resnet50.out --output=logs/run_resnet50.out run_script.sh resnet50 # Submit a job
squeue -u zzhou82 # Print the current job list
myjobs # Print the current job list in detail
myjobs | wc -l # Count the total number of current jobs (maximum 1000 jobs allowed per users)
scancel *** (JOBID)
scontrol show job JOBID # Check the detailed running script for JOBID
```

##### An example of shell scripts (in run_script.sh)
```
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --mem-per-cpu 50000
##SBATCH -p physicsgpu1
##SBATCH -p sulcgpu2
##SBATCH -p mrlinegpu1
#SBATCH -p asinghargpu1
##SBATCH -p sulcgpu1
##SBATCH -p cidsegpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -t 9-23:59:59
##SBATCH -o slurm.%j.${1}.out
##SBATCH -e slurm.%j.${1}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zzhou82@asu.edu

module load tensorflow/1.8-agave-gpu
module unload python/.2.7.14-tf18-gpu

# autoencoder
python3.6 -W ignore Genesis_ImageNet_2D.py --note autoencoder --rescale_rate 0.1 --flip_rate 0.01 --rotation_rate 0.01 --transpose_rate 0.01 --gamma_rate 0.0 --paint_rate 0.0 --outpaint_rate 0.8 --local_rate 0.0 --nonlinear_rate 0.0 --twist_rate 0.0 --blur_rate 0.0 --arch Unet --decoder upsampling --backbone $1 --batch_size 24 --input_rows 224 --input_cols 224 --input_deps 3 --crop_rows 512 --crop_cols 512 --crop_deps 3 --steps_per_epoch 5000 --worker 4 --init finetune --nb_class 3 --verbose 1 --optimizer sgd --save_samples png --data /home/zzhou82/zongwei.zhou/Models-Genesis/ImageNet/ILSVRC2012
```
More examples and explanations can be found at https://rcstatus.asu.edu/agave/howto/gpu.php
```
#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 1                        # number of CPU cores to reserve on this compute node

#SBATCH -p gpu                      # This is the recommended default partition
###SBATCH -p cidsegpu1              # Uncomment this line to specifically callout the cidsegpu1 partition instead

#SBATCH -q wildfire                 # Run job under wildfire QOS queue

#SBATCH --gres=gpu:2                # Request two GPUs

#SBATCH -t 0-12:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=myemail@asu.edu # send-to address
```

Check the graphical view of Agave here: https://rcstatus.asu.edu/agave/smallstatus.php

##### Shared folder with unlimited space in ASU GPU:
zzhou82@agave.asu.edu:/scratch/zzhou82/

##### Transfer files to Google Drive
- Documentation: https://asurc.atlassian.net/wiki/spaces/RC/pages/349175850/Configuring+Globus+Collections


# Access Bridge AI GPU
- Credit to [Vatsal Sodha](https://github.com/vatsal-sodha)

##### 1. Create an account at 
https://portal.xsede.org/psc-bridges

##### 2. Set up the password for PSC at 
https://apr.psc.edu/autopwdreset/autopwdreset.html

##### 3. Good to go!
```bash
ssh -Y zzhou82@bridges.psc.xsede.org
```
List of all modules: https://www.psc.edu/resources/software

##### 4. Start a job
```bash
sbatch --error=logs/bms.out --output=logs/bms.out run_script.sh
```

##### Check the summary of projects
```bash
projects
```

##### An example of shell scripts (in run_script.sh)
```bash
#!/usr/bin/env bash

#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH -t 36:00:00
#SBATCH -A bc5phlp

##SBATCH --ntasks-per-node=2
##SBATCH --gres=gpu:volta32:1

#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:volta16:1

#SBATCH --mail-type=ALL                # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=zzhou82@asu.edu     # send-to address

source /etc/profile.d/modules.sh

module load AI/anaconda3-5.1.0_gpu.2018-08
source activate mask3


python -W ignore main.py --apps bms --run 1 --cv 1 --suffix genesis-ct --task segmentation --subsetting None --data /pylon5/bc5phhp/zzhou82/BraTS --mode train
```


##### Transfer the dataset into /pylon5/bc5phhp/zzhou82
```
scp config.py zzhou82@data.bridges.psc.edu:/pylon5/bc5phhp/zzhou82/
```

##### Transfer keras pre-trained imagenet models into BridgeAI machine
```
scp /home/zongwei/.keras/models/*.h5 zzhou82@data.bridges.psc.edu:/home/zzhou82/.keras/models/
```
It has nearly unlimited space
```
[zzhou82@login006 zzhou82]$ df -h /pylon5/bc5phhp/zzhou82
Filesystem                Size  Used Avail Use% Mounted on
10.4.112.79@o2ib:/pylon5  7.8P  6.1P  1.7P  79% /pylon5
```

# Remotely Jupyter notebook
##### 1. Open Cisco AnyConnect Secure Mobility Client
Type in: `sslvpn.asu.edu` and click `Connect`

##### 2. Type in user name (asu email) and password

##### 3. Set up a Jupyter notebook in the remote machine
```bash
ssh zongwei@t2-host.hfc.dhcp.asu.edu -X
pip install jupyter # install jupyter notebook
nohup jupyter notebook --no-browser --notebook-dir='/mnt/dfs/zongwei' --port=8881 > /home/zongwei/jupyter.log &
nohup jupyter notebook --no-browser --notebook-dir='/data/zzhou82' --port=8885 > /home/zzhou82/jupyter.log &
```

##### 4. Get the link
```bash
cat /home/zongwei/jupyter.log
```

```
[I 00:02:03.886 NotebookApp] Serving notebooks from local directory: /mnt/dfs/zongwei
[I 00:02:03.886 NotebookApp] The Jupyter Notebook is running at:
[I 00:02:03.886 NotebookApp] http://localhost:8881/?token=7cc992537c1209286361db906cff0128670858e0725925f5
[I 00:02:03.886 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 00:02:03.891 NotebookApp]
 To access the notebook, open this file in a browser:
 file:///run/user/1201/jupyter/nbserver-25432-open.html
 Or copy and paste one of these URLs:
http://localhost:8881/?token=7cc992537c1209286361db906cff0128670858e0725925f5
[W 00:02:09.161 NotebookApp] 404 GET /api/kernels/4c16c34c-ae91-45c0-a5d3-3f6a3be0dac4/channels?session_id=ea67fd77aac4414184755e7c14351c3a (127.0.0.1): Kernel does not exist: 4c16c34c-ae91-45c0-a5d3-3f6a3be0dac4
[W 00:02:09.188 NotebookApp] 404 GET /api/kernels/4c16c34c-ae91-45c0-a5d3-3f6a3be0dac4/channels?session_id=ea67fd77aac4414184755e7c14351c3a (127.0.0.1) 41.81ms referer=None
[I 00:02:52.752 NotebookApp] Kernel started: 4cbe272f-abf8-4751-af89-392f0c7b0890
[I 00:02:53.058 NotebookApp] Adapting to protocol v5.1 for kernel 4cbe272f-abf8-4751-af89-392f0c7b0890
```

##### 5. Copy the link and paste it to your own laptop browser. In the previous example, the link is http://localhost:8881/?token=7cc992537c1209286361db906cff0128670858e0725925f5

##### 6. Set up in your laptop terminal
```bash
ssh -N -f -L localhost:8881:localhost:8881 zongwei@t2-host.hfc.dhcp.asu.edu
ssh -N -f -L localhost:8885:localhost:8885 zzhou82@ccvl25.ccvl.jhu.edu
```

##### 7. Remember to kill the process in both client (laptop) and server (t2-host.hfc.dhcp.asu.edu) when finished.
```bash
lsof -ti:8881| xargs kill -9
```

# Use Github in Linux
```bash
git clone https://github.com/MrGiovanni/ModelsGenesis.git
```
##### Push the local changes on the website
```bash
git add .
git commit -am "make some comments "
git push origin master
```

# Mount remote machine to MacBook
- Credit to [Vatsal Sodha](https://github.com/vatsal-sodha)

Download and install FUSE from https://github.com/osxfuse/osxfuse/releases/tag/osxfuse-3.10.4
```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> /Users/$USER/.zprofile
eval $(/opt/homebrew/bin/brew shellenv)
brew install osxfuse
brew install sshfs
sshfs -o reconnect -o follow_symlinks -o volname=dfs -o IdentityFile=~/.ssh/id_rsa zongwei@t2-host.hfc.dhcp.asu.edu:/mnt/dfs/ /Users/zongwei.zhou/Documents/dfs/ -o volname=dfs
sshfs -o reconnect -o follow_symlinks -o volname=asu -o IdentityFile=~/.ssh/id_rsa zzhou82@agave.asu.edu:/home/zzhou82/zongwei.zhou/ /Users/zongwei.zhou/Documents/asu/ -o volname=asu
sshfs -o reconnect -o follow_symlinks -o volname=jhu -o IdentityFile=~/.ssh/id_rsa zzhou82@ccvl25.ccvl.jhu.edu:/data/zzhou82/ /Users/zongwei.zhou/Documents/jhu/ -o volname=jhu
diskutil unmount force /Users/zongwei.zhou/Documents/dfs/
diskutil unmount force /Users/zongwei.zhou/Documents/asu/
diskutil unmount force /Users/zongwei.zhou/Documents/jhu/
```

# Miniconda
##### Check current existing conda environments
```bash
conda env list
```
It returns information as following:
```
conda environments:
#
caffe /home/zongwei/miniconda3/envs/caffe
root * /home/zongwei/miniconda3
```

##### Create a new conda environment named "env_name"
```bash
conda create -n env_name python=3.6 -y
```

##### Delete a conda environment named "env_name"
```bash
conda env remove -n env_name
```

##### Activate "env_name" environment (independent from other conda environments). So if you mess up one environment, it would not affect on other environment
```bash
source activate env_name
```

##### Deactivate current environment
```bash
source deactivate
```

##### Install the packages you may need
```bash
conda install matplotlib tqdm keras==2.2.4 tensorflow-gpu==1.13.1
pip install opencv-python pillow tqdm scikit-image sklearn photutils simpleitk nibabel pandas
```

# Random things about Ubuntu
- Give sudoer permission: ```sudo su```
- Check GPU information: ```nvidia-smi -l 1```
- Check CPU information: ```htop```
- Show system path: ```echo $PATH | tr ":" "\n" | nl```
- Install .deb file: ```sudo dpkg -i teamviewer_12.0.71510_i386.deb```
- Check the size of file: ```ls -lh filename```
- Check the disc space used and left in the current address: ```df -h .```
- Compress a directory/file: ```tar -czvf name-of-archive.tar.gz /path/to/directory-or-file```
- Compress many folders: ```for name in Covid19 dsb2018; do tar -czvf $name.tar.gz $name; done```
- Extract an archive: ```tar -xzvf archive.tar.gz```
- Delete files forever: ```/bin/rm -rf /mnt/local/zongwei/filename```
- Grant permission to a folder and files: ```chmod -R 777 *```
- Check the size of a directory: ```du -sh /home/zongwei```
- Specify the GPU for a job: ```CUDA_VISIBLE_DEVICES=0 python```
- Count folders in a directory: ```ls -dq 01_Multi-Atlas_Labeling_img00* | wc -l```
- Clear Google Drive cache: ```rm -rf ~/Library/Application\ Support/Google/DriveFS/```
- Create shortcut: ```ln -s ~/Target/Folder ~/Desktop```
- Use 'll' in Mac OS: ```alias ll='ls -lGaf'```
- Check the storage: ```du -B1 -cs /Volumes/GoogleDrive/*```
- Submit many jobs: ```while [ $(myjobs | wc -l) < 1000 ]; do echo 'hi'; sleep 5s; done```
- Delete jobs: ```fuser -k /dev/nvidia*```
- Kill all processes for the user: ```pkill -u zzhou82```
