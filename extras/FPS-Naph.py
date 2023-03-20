from ast import parse
import os, sys, time
import argparse
import tqdm

import numpy as np
from skcosmo.sample_selection import FPS

def read_sf(SFP, ENG):
    #number_SF = int() #TODO : How to input the number of sf as a variable

    se = np.loadtxt(SFP, usecols = range(1, 43))
    en = np.loadtxt(ENG)

    #assert se.shape[0] == en.shape[0]

    return se, en
    
def sampling(se):
    selector = FPS(
    n_to_select = 10000,
    progress_bar = True,
    score_threshold = 1E-12,
    full = False,
    initialize = 'random')

    selector.fit(se)

    Xr = selector.transform(se)
    idx = np.array(selector.get_support(indices = True))

    return idx, Xr

def write_FPS(idx, output_file_name):
    input_file = open('input.data', 'r')
    output_file = open(output_file_name, 'w')

    frame = 0

    n_atoms = 144
    atomic_sym = []

    flag = True

    pos = np.zeros((n_atoms, 3))
    force = np.zeros((n_atoms, 3))

    c1 = 0.0
    c2 = 0.00000000

    while True:

        line = input_file.readline().rstrip()   #this will always skip one line

        if 'lattice' in line:
            lat = np.zeros((3,3))
            #Lattice vectors
            lat[0, 0] = float(line.split()[1])
            lat[0, 1] = float(line.split()[2])
            lat[0, 2] = float(line.split()[3])

            line = input_file.readline().rstrip()   #moving to next line
            lat[1, 0] = float(line.split()[1])
            lat[1, 1] = float(line.split()[2])
            lat[1, 2] = float(line.split()[3])

            line = input_file.readline().rstrip()   #moving to next line
            lat[2, 0] = float(line.split()[1])
            lat[2, 1] = float(line.split()[2])
            lat[2, 2] = float(line.split()[3])
        
        elif 'atom' in line:

            for i in range(n_atoms):    #atomic positions and forces
                pos[i, 0] = float(line.split()[1])
                pos[i, 1] = float(line.split()[2])
                pos[i, 2] = float(line.split()[3])
                
                if flag == True:
                    atomic_sym.append(line.split()[4])

                force[i, 0] = float(line.split()[7])
                force[i, 1] = float(line.split()[8])
                force[i, 2] = float(line.split()[9])

                if i == n_atoms - 1:    #important: Otherwise skips two lines
                    flag = False
                    break
                else:
                    line = input_file.readline().rstrip()

        elif 'energy' in line:  #Energy
            energy = float(line.split()[1])

        elif 'charge' in line:  #Charge
            charge = float(line.split()[1])
            line = input_file.readline().rstrip()
        
        if 'end' in line:   #Writing each frame
            
            if frame in idx:
                output_file.write('begin\n')

                output_file.write('lattice\t{: .10f}  {: .10f}  {: .10f}  \n'.format(lat[0, 0], lat[0, 1], lat[0, 2]))
                output_file.write('lattice\t{: .10f}  {: .10f}  {: .10f}  \n'.format(lat[1, 0], lat[1, 1], lat[1, 2]))
                output_file.write('lattice\t{: .10f}  {: .10f}  {: .10f}  \n'.format(lat[2, 0], lat[2, 1], lat[2, 2]))
                for i in range(n_atoms):
                    output_file.write('atom\t{: .10f}  {: .10f} {: .10f}  {}  {: .1f}  {: .10f}  {: .10f}  {: .10f}  {: .10f}\n'.format(pos[i, 0], pos[i, 1], pos[i, 2], atomic_sym[i], c1, c2, force[i, 0], force[i, 1], force[i, 2] ))

                output_file.write('energy\t{: .10f}\n'.format(energy))
                output_file.write('charge\t{: .10f}\n'.format(charge))
                output_file.write('end\n')
            
            frame += 1

        if not line:
            break
            
    input_file.close()
    output_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = """ Farthest Point Sampling """)

    #parser.add_argument('input-data', type= str)
    parser.add_argument('-o', type = str, default= 'out')
    parser.add_argument('-n', type = int, default= '', help= 'Number of output samples')

    args = parser.parse_args()

    output_file_name = args.o
    no_of_output = args.n

    read_sf(SFP = 'function.cleaned.data', ENG = 'energy.data')
    sampling(se)
    write_FPS(idx, output_file_name)