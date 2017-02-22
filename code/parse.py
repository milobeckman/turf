
## parse.py
## this file reads a data file (tab-separated vote results) into an Election object

from __future__ import division
import numpy as np


class Election:
    def __init__(self, name):
        self.name = name
        self.candidates = set()
        self.votes = {}
    
    # add the specified data point to this Election
    def add_vote_for_candidate(self, county, candidate, numvotes):
        self.votes.setdefault(county, {})[candidate] = numvotes
        self.candidates.add(candidate)
    
    # returns a 2d array of votes results, as well as county and candidate lists
    def get_results(self, dtype):
        counties = sorted(self.votes.keys())
        candidates = sorted(self.candidates)
        
        # generate the vote array, normalizing the sum of votes for each county to 1
        vote_array = np.array([[self.votes[t].get(k,0) / sum(self.votes[t].values()) for k in candidates] for t in counties], dtype=dtype)
        
        return vote_array, counties, candidates


# return an Election object from the provided data file
def get_election_from_data(filename):
    
    # this should obviously not be hardcoded in
    e = Election("GOP Super Tuesday 2016")
    
    # if data contains state information (full row whose only entry is state name) we store (county,state) pairs
    separate_state = False

    candidates = None
    cur_state = None
    cur_county = None
    
    first_line = True
    
    county_idx = 1 if separate_state else 0
    vote_idx = 2 if separate_state else 1
    
    for line in open(filename):
        line = line.rstrip("\n").split("\t")
        if first_line:
            
            # check if data contains state information
            if line[1] == "":
                separate_state = True
            county_idx = 1 if separate_state else 0
            vote_idx = 2 if separate_state else 1
            
            first_line = False
            candidates = line[vote_idx:]
            continue
        
        # new state
        if separate_state and line[0] != "":
            cur_state = line[0]
            
        cur_county = line[county_idx]
        for candidate, vote_str in zip(candidates, line[vote_idx:]):
            
            # this has to do with specifics of the raw data we're using
            if vote_str == "#N/A":
                continue
            if candidate == "Uncommitted":
                continue
            vote = float(vote_str)
            if vote != 0:
                county_id = (cur_state, cur_county) if separate_state else cur_county
                e.add_vote_for_candidate(county_id, candidate, vote)
    
    return e
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    