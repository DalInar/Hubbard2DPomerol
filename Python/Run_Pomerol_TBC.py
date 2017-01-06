import numpy
import math
import sys
import os
import h5py
import json
import subprocess as sp
import argparse

def get_phases():
	L = 7
	IncEndPt = True
	MaxPhase = 2*math.pi

	theta = numpy.linspace(0, MaxPhase, L, IncEndPt)
	thetas = [(x,y) for x in theta for y in theta]
	return thetas
	
def get_command(beta, U, mu, phase_x, phase_y):
	cmd = "/home/oryx/ClionProjects/Hubbard2DPomerol/build/Hubbard2DCrossPomerol -b "+str(beta)+" -U "+str(U)+" --mu "+str(mu)+" --phase_x "+str(phase_x)+" --phase_y "+str(phase_y)+" --calcgf --wf 256"
	#cmd = "/home/oryx/ClionProjects/Hubbard2DPomerol/build/Hubbard2DPomerol -b "+str(beta)+" -U "+str(U)+" --mu "+str(mu)+" --phase_x "+str(phase_x)+" --phase_y "+str(phase_y)+" -x 2 -y 2 --calcgf"
	return cmd
	
def read_energy():
	f=open('Energy.dat',"r")
	energy=numpy.loadtxt(f)
	f.close()
	
	return float(energy)

def main():
	print("Hi! I run Pomerol for a bunch of TBC phases and collect the energies.")
	parser = argparse.ArgumentParser(description='Run Pomerol for different TBC')
	parser.add_argument('beta', type=float, help='Inverse temperature')
	parser.add_argument('U', type=float, help='Interaction strength')
	parser.add_argument('mu', type=float, help='Chemical Potential')
	parser.add_argument('--run', type=bool, default=False, help='Chemical Potential')
	args=parser.parse_args()
	
	beta = args.beta
	U = args.U
	mu=args.mu
	run = args.run
	
	log = open("log.dat","w")
	log.write("beta="+str(beta)+"\n")
	log.write("U="+str(U)+"\n")
	log.write("mu="+str(mu)+"\n")
	log.write(get_command(beta,U,mu,0,0))
	log.close()
	
	phases = get_phases()
	
	data = []
	for phase in phases:
		phase_x = phase[0]
		phase_y = phase[1]
		directory = "Phase_x_"+str("%0.2f" % phase_x)+"_Phase_y_"+str("%0.2f" % phase_y)
		print(directory)
		print(get_command(beta,U,mu,phase_x,phase_y))
		command = "mkdir -p ./"+directory
		sp.call(command, shell=True)
		os.chdir(directory)
		if(run):
			sp.call(get_command(beta,U,mu,phase_x,phase_y),shell=True)
		energy=read_energy()
		d={"PHASE_0":phase_x,"PHASE_1":phase_y,"Energy":energy}
		data.append(d)
		os.chdir("..")
		
	energy_log = open("Energy_Log.json","w")
	energy_log.write(json.dumps(data))
	energy_log.close()
	
if __name__=="__main__":
	main()
