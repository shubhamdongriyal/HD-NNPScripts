from ast import parse
import os, sys, time
import argparse
import tqdm

import numpy as np
from skcosmo.sample_selection import FPS

def read_sf(SFP, ENG):

    print('Reading the input.data and function.data files\n\n')

    se = np.loadtxt(SFP, usecols = range(1, NoSF + 1))
    se = se.reshape((-1,NoAtoms * NoSF))
    
    en = np.loadtxt(ENG)

    assert se.shape[0] == en.shape[0]

    print('Reading Done !!\n')
    return se, en
    
def sampling(se):

    print('Now start the FPS Algorithm\n\n')

    selector = FPS(
    n_to_select = no_of_output,
    progress_bar = True,
    score_threshold = 1E-12,
    full = False,
    initialize = 'random')

    selector.fit(se)

    Xr = selector.transform(se)
    idx = np.array(selector.get_support(indices = True))

    print(f'FPS Done!!!!\n\nNow you have {len(idx)} diverse samples\n\n')
    return idx, Xr

def write_FPS(idx, InData, output_file_name):

    print('Writing a new input.data file which only contains the diverse samples\n\n')
    input_file = open(InData, 'r')
    output_file = open(output_file_name, 'w')

    frame = 0

    n_atoms = NoAtoms
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

    print('You are done here!!')

def main():

    
    SFP = 'function.cleaned.data'
    ENG = 'energy.data'
    InData = 'input.data'

    se, en = read_sf(SFP , ENG)
    idx, Xr = sampling(se)
    write_FPS(idx, InData, output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = """ Farthest Point Sampling """)

    #parser.add_argument('input-data', type= str)
    parser.add_argument('-o', type = str, default= 'out')
    parser.add_argument('-n', type = int, default= '', help= 'Number of output samples')

    args = parser.parse_args()

    output_file_name = args.o
    no_of_output = args.n
    
    print('Use do-FarthestPointSampling.sh script to run this code')
    print('\n\nThis code is to compute the diverse samples using the Farthest Point Sampling (FPS)\n\n')
    
    NoAtoms = int(input(' How many number of ATOMS in a given sample/frame:\t'))
    NoSF = int(input(' Please provide the number of SYMMETRY FUNCTIONS:\t'))
    main()
