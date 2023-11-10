import subprocess

#Arash Ansari

hmm_script = './HMM.py'


print("Problem 2: GENERATE\n--------------------")
arguments = ["python", "hmm.py", "--generate", "20"]

subprocess.run(arguments)

print("\nProblem 2: FORWARD\n-------------------")
arguments = ["python", "hmm.py", "--forward", "ambiguous_sents.obs"]

subprocess.run(arguments)

print("\nProblem 2: VITERBI\n-------------------")
arguments = ["python", "hmm.py", "--viterbi", "ambiguous_sents.obs"]

subprocess.run(arguments)


print("\nProblem 3: ALARM\n-------------------")
arguments = ["python", "alarm.py"]

subprocess.run(arguments)

print("\nProblem 3: CARNET\n-------------------")
arguments = ["python", "carnet.py"]

subprocess.run(arguments)
