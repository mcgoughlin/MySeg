import torch
import sys
import os
import subprocess

gpus = int(sys.argv[1])
script = str(sys.argv[2])
data_name = str(sys.argv[3])
spacing = str(sys.argv[4]).split(",")
fold = str(sys.argv[5]).split(",")

for dev in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(dev))

os.chdir("/bask/homes/r/ropj6012/segmentation/plain_ovseg")
print(os.listdir(os.getcwd()))

run_string = ""
processes = []
for gpu in range(gpus):
    spac = spacing[gpu%len(spacing)]
    f = fold[gpu%len(fold)]

    if gpu == gpus-1:
        run_string = "python "+script+" "+data_name+" "+str(spac)+" "+str(f)
    else:
        run_string = "python "+script+" "+data_name+" "+str(spac)+" "+str(f) +" &"

    print('#####################')
    print(run_string)
    print('#####################')
    with torch.cuda.device(gpu):
        processes.append(subprocess.Popen(run_string, shell=True,stdout=subprocess.PIPE))

#makes sure all training finishes before ending scripts - no ending early
for process in processes:
    process.wait()