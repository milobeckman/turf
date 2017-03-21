
## gen-test-case.py
## this script reads a file with candidates positions and strengths, and writes a test case with this as a zero-error solution

from __future__ import division
import numpy as np
import sys

import optimize


# returns in string from the vote returns for a county 
def row_for_county(candidate_strengths, candidate_positions, county_position, county_id):
    saliences = []
    for i in range(len(candidate_strengths)):
        saliences += [optimize.salience_for_candidate_and_county(candidate_strengths[i], candidate_positions[i], county_position)]
    saliences = [c/sum(saliences) for c in saliences]
    
    row = county_id + "\t"
    for c in saliences:
        row += str(c) + "\t"
    
    return row

# returns a list of all county vote returns over a lattice of counties
def list_of_all_rows(candidate_strengths, candidate_positions, min_pos, max_pos, step_size):
    
    # this only works for two dimensions
    rows = []
    
    x,i = min_pos,0
    while x <= max_pos:
        y,j = min_pos,0
        while y <= max_pos:
            county_position = np.array([x,y])
            county_id = "(" + str(x) + "," + str(y) + ")"
            rows += [row_for_county(candidate_strengths, candidate_positions, county_position, county_id)]
            
            j += 1
            y += step_size
        i += 1
        x += step_size
    
    return rows

# writes the list of rows to the specified file with the specified candidate names
def write_rows_to_file(rows, candidate_ids, filename):
    header = ""
    for cid in candidate_ids:
        header += "\t" + cid
    
    f = open(filename, "w")
    f.write(header + "\n")
    for row in rows:
        f.write(row + "\n")

# reads a text file with candidate strengths and positions
def read_candidates_from_file(filename):

    # the file looks like this:
    '''
sw	1	-1	-1
se	1	1	-1
nw	1	-1	1
NE!	2	1	1   
    '''
    # the first column is names, the second column is strengths, the rest is (x,y) posns
    
    candidate_strengths, candidate_positions, candidate_ids = [],[],[]
    rows = open(filename, "r").readlines()
    for row in rows:
        candidate_info_list = row.split("\t")
        candidate_ids += [candidate_info_list[0]]
        candidate_strengths += [float(candidate_info_list[1])]
        candidate_positions += [np.array([float(candidate_info_list[2]),float(candidate_info_list[3])])]
    
    return candidate_ids, candidate_strengths, candidate_positions

# takes in an input file (see format above) and writes the test case
def main():
    candidate_ids, candidate_strengths, candidate_positions = read_candidates_from_file(sys.argv[1])
    rows = list_of_all_rows(candidate_strengths, candidate_positions, -3., 3., 0.5) # this should really be a command-line argument
    write_rows_to_file(rows, candidate_ids, sys.argv[2])
    
    



if __name__ == '__main__':
    main()