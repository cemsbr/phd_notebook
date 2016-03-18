strong_scaling2
---------------
app: top_contributors.py
input: 1 GB
VMs: 1, 2, 3
replications: 1, 2, 3
repetitions: 10
experiment number: 6

one_vm
------
app: top_contributors.py
input: 128, 256, 512, 1024 MB
VMS: 1
repetitions: 10
experiment number: 5

strong_scaling1
---------------
app: top_contributors.py
input: 1 GB
VMs: 1, 2, 4
replications: 1, 2, 4
repetitions: 10
experiment number: 4

weak_scaling
------------
app: top_contributors.py
input: 1, 2, 4, 8, 16, 32, 45 GB
VMs:   1, 2, 4, 8, 16, 32, 45
repetitions: 30
Experiment number: 2
