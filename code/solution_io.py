
## solution_io.py
## this file reads and writes solutions (current best arrangements of candidates and counties) from/to a local file

import os
import sys
import json
import numpy as np

from config import config
import parse


# read solution from filename
def read_solution(filename):
    candidate_strengths = None
    candidate_positions = None
    voter_positions = None
    error = None
    step_no = 0
    
    # read whatever data exists from file
    if os.path.exists(config.solution_file):
        state = json.loads(open(config.solution_file).read())
        
        if "candidate_strengths" in state:
            candidate_strengths = np.array(state["candidate_strengths"], dtype=config.dtype)
        if "candidate_positions" in state:
            candidate_positions = np.array(state["candidate_positions"], dtype=config.dtype)
        if "voter_positions" in state:
            voter_positions = np.array(state["voter_positions"], dtype=config.dtype)
        if "error" in state:
            error = state["error"]
        if "step_no" in state:
            step_no = state["step_no"]

    return candidate_positions, candidate_strengths, voter_positions, error, step_no

def read_latest_positions(filename):
    positions_filename = filename[:-4] + "_positions.txt"
    
    # read whatever data exists from file
    if not os.path.exists(positions_filename):
        return None, None, None, None, 0
    
    position_lines = [x.strip() for x in open(positions_filename).readlines()]
    header = position_lines[0]
    latest = position_lines[-1]
    
    header_cols = header.split("\t")
    latest_data = [float(x) for x in latest.split("\t")]
    
    candidate_strengths = []
    candidate_positions = []
    voter_positions = []
    error = None
    step_no = len(position_lines)-1
    
    col = 0
    
    # parsing candidate (x,y)s
    while (header_cols[col][-2:] != "_s"):
        candidate_positions += [[latest_data[col], latest_data[col+1]]]
        col += 2
    
    # parsing candidate strengths
    while (header_cols[col][-2:] == "_s"):
        candidate_strengths += [latest_data[col]]
        col += 1
    
    # parsing county (x,y)s
    while (col < len(header_cols)):
        voter_positions += [[latest_data[col], latest_data[col+1]]]
        col += 2
    
    return np.array(candidate_positions), np.array(candidate_strengths), np.array(voter_positions), error, step_no

# write probabilities to filename
def write_probabilities(filename, prior_vector, probability_vector, step_no):

    prior_strength, prior_candidate_location, prior_county_location, prior_spread = prior_vector
    prior, likelihood, posterior = probability_vector

    filename = filename.replace('.txt','_probabilities.txt')
    if os.path.exists(filename) == False:
        header = [  'step_no','posterior','likelihood','prior',
                    'prior_s','prior_candidate_xy','prior_county_xy','prior_spread' ] 

        open(filename, "a+").write('\t'.join(header)+'\n')

    line = '\t'.join([str(x) for x in [ step_no,
                                        posterior,
                                        likelihood,
                                        prior,
                                        prior_strength,
                                        prior_candidate_location,
                                        prior_county_location,
                                        prior_spread]])

    open(filename, "a+").write(line+'\n')


# write positions to filename
def write_positions(filename, candidate_positions, candidate_strengths, county_positions, step_no=-1):
    filename = filename.replace('.txt','_positions.txt')

    if os.path.exists(filename) == False:
        election = parse.get_election_from_data(config.data_file)
        actual_votes, counties, candidates = election.get_results(config.dtype)

        candidate_header = [ candidate+'_'+ax for candidate in candidates for ax in ['x','y'] ]
        strength_header = [ candidate+'_s' for candidate in candidates ]
        county_header = [ county+'_'+ax for county in counties for ax in ['x','y'] ]
        header = candidate_header + strength_header + county_header

        open(filename, "a+").write('\t'.join(header)+'\n')

    candidate_line = '\t'.join(['\t'.join([str(x) for x in pos]) for pos in candidate_positions])
    strength_line = '\t'.join([str(x) for x in candidate_strengths])
    county_line = '\t'.join(['\t'.join([str(x) for x in pos]) for pos in county_positions])

    line = '\t'.join([candidate_line, strength_line, county_line])
    open(filename, "a+").write(line+'\n')


def write_solution(filename, candidate_positions, candidate_strengths, voter_positions, total_error,step_no):
  state = {
    "error": total_error,
    "candidate_strengths": candidate_strengths.tolist(),
    "candidate_positions": candidate_positions.tolist(),
    "voter_positions": voter_positions.tolist(),
    "step_no": step_no
  }

  open(filename, "w").write(json.dumps(state, sort_keys=True))

# # write solution to filename
# def write_probabilities(filename, prior, likelihood, posterior, step_no=-1):
#     state = {
#         "prior": prior,
#         "likelihood": likelihood,
#         "posterior": posterior,
#         "step_no": step_no
#     }
#     filename = filename.replace('.txt','_probabilities.txt')
#     open(filename, "a+").write(json.dumps(state, sort_keys=True)+'\n')


# def write_positions(filename, candidate_positions, candidate_strengths, county_positions, step_no=-1):
#     state = {
#         "candidate_strengths": candidate_strengths.tolist(),
#         "candidate_positions": candidate_positions.tolist(),
#         "voter_positions": county_positions.tolist(),
#         "step_no": step_no
#     }
#     filename = filename.replace('.txt','_positions.txt')
#     open(filename, "a+").write(json.dumps(state, sort_keys=True)+'\n')
