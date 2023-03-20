import argparse
from ase.io import read, write

def main(inputfile):
    FILE = read(inputfile, format = 'aims')

    return write(outputfile, FILE, format = 'lammps-data')

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = """"This script is to convert the aims geometry.in file to lammps structure file""")

    parser.add_argument('-i', type = str)
    parser.add_argument('-o', type = str)

    args = parser.parse_args()
    inputfile = args.i
    outputfile = args.o
    print('\n\n************* This script is to convert the aims geometry.in file to lammps structure file *************')
    main(inputfile)
