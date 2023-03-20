import argparse
from ase.io import iread

def main(inputfile):

#aims output file
    trj = iread(inputfile, format='vasp-xml')
    #trj = ase.read_vasp_xml(filename=inputfile)

#create input file for n2p2
    print('*************** Writing the add-input.data trajectory file ********************')
    with open('add-input.data', 'w', encoding='utf-8') as x:
        for s in trj:
        
            x.write('begin\n')
            
            #uncomment below lines if it containes lattice vectors

            x.write('lattice\t'+str('\t'.join(str(f) for f in s.get_cell()[0]))+'\n')
            x.write('lattice\t'+str('\t'.join(str(f) for f in s.get_cell()[1]))+'\n')
            x.write('lattice\t'+str('\t'.join(str(f) for f in s.get_cell()[2]))+'\n')
            
            for i in range(len(s.get_positions())):
                x.write('atom\t'+str(s.get_positions()[i][0])+'\t'+ str(s.get_positions()[i][1])+'\t'+ str(s.get_positions()[i][2])+ '\t'+str(s.get_chemical_symbols()[i])+ '\t'+ str(s.get_initial_charges()[i])+ '\t' + '0.00000000'+ '\t' + str(s.get_forces()[i][0])+ '\t' + str(s.get_forces()[i][1])+ '\t' + str(s.get_forces()[i][2]) +'\n')
        
            x.write('energy\t'+str(s.get_potential_energy())+'\n')
            x.write('charge\t0.00000000\n')
            x.write('end\n')

        x.close()

if __name__ =="__main__":
    parser = argparse.ArgumentParser( description = """This script is to convert the aims trajectory file to input.data file of n2p2 """)

    parser.add_argument('-i', type = str)

    args = parser.parse_args()
    inputfile = args.i

    main(inputfile)
