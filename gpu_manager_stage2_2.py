import torch
import sys
import os
import subprocess

gpus = int(sys.argv[1])
script = str(sys.argv[2])
data_name = str(sys.argv[3])
og_spacing1 = str(sys.argv[4]).split(",")
og_spacing2 = str(sys.argv[5]).split(",")
target_spacing = str(sys.argv[6]).split(",")
fold = str(sys.argv[7]).split(",")

for dev in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(dev))

os.chdir("/bask/homes/r/ropj6012/segmentation/plain_ovseg")
print(os.listdir(os.getcwd()))

run_string = ""
processes = []
for gpu in range(gpus):
    spac1 = og_spacing1[gpu%len(og_spacing1)]
    spac2 = og_spacing2[gpu%len(og_spacing2)]
    tgt = target_spacing[gpu%len(target_spacing)]
    f = fold[gpu%len(fold)]

    if gpu == gpus-1:
        run_string = "python "+script+" "+data_name+" "+str(spac1)+" "+str(spac2)+" "+str(tgt)+" "+str(f)
    else:
        run_string = "python "+script+" "+data_name+" "+str(spac1)+" "+str(spac2)+" "+str(tgt)+" "+str(f) +" &"

    print('#####################')
    print(run_string)
    print('#####################')
    with torch.cuda.device(gpu):
        processes.append(subprocess.Popen(run_string, shell=True,stdout=subprocess.PIPE))

#makes sure all training finishes before ending scripts - no ending early
for process in processes:
    process.wait()