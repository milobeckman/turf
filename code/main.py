
## main.py
## this file loads the data and uses the specified implementation to (try to) find the optimal arrangement

from __future__ import division
import numpy as np

from config import *
import solution_io
import parse
import optimize
import visualize


# read command line arguments and run the appropriate optimization
def main():
    
    # read and report command line arguments
    parse_args()
    print("Running with the following config:")
    for k, v in sorted(vars(config).items()):
        if k is not "rand_order_of_counties":
            print(k, "=", repr(v))

    # parse the data file
    election = parse.get_election_from_data(config.data_file)
    actual_votes, counties, candidates = election.get_results(config.dtype)

    # if we're using only some of the counties, choose them
    if config.num_counties is not None:
        actual_votes = np.array([actual_votes[i] for i in config.rand_order_of_counties[:config.num_counties]], dtype=config.dtype)
        counties = np.array([counties[i] for i in config.rand_order_of_counties[:config.num_counties]])

    # if we're using only some of the candidates, choose them
    if config.candidates is not None:
        actual_votes = np.array([[vote[i] for i in config.candidates] for vote in actual_votes], dtype=config.dtype)
        candidates = np.array([candidates[i] for i in config.candidates])

    # if config.plot_trace_positions is True or config.plot_trace_posteriors is True:
    #     config.write_every_best_solution == True

    # normalize the votes to sum to 1
    for i in range(len(actual_votes)):
        actual_votes[i] /= sum(actual_votes[i])
    for i in range(len(actual_votes)):
        assert round(sum(actual_votes[i]), 5) == 1.0, (sum(actual_votes[i]), actual_votes[i])

    print(len(counties), "counties")
    print(len(candidates), "candidates")
    print(candidates)

    # initialize everything
    candidate_strengths = None
    candidate_positions = None
    county_positions = None
    total_error = None
    step_no = 0

    # read from solution file unless --no-read-solution is specified
    if config.read_solution:
        candidate_positions, candidate_strengths, county_positions, total_error, step_no = solution_io.read_solution(config.solution_file)
    if candidate_strengths is not None:
        assert candidate_strengths.shape == (len(candidates), ), (candidate_strengths.shape, (len(candidates), ))
    if candidate_positions is not None:
        assert candidate_positions.shape == (len(candidates), config.dimensions), (candidate_positions.shape, (len(candidates), config.dimensions))
    if county_positions is not None:
        assert county_positions.shape == (len(counties), config.dimensions), (county_positions.shape, (len(counties), config.dimensions))

    # run the appropriate optimization
    
    if config.method == "single_inner":
        county_positions = np.zeros((len(counties), config.dimensions), dtype=config.dtype)
        county_positions = optimize.find_minimal_error_positions_for_all_counties(candidate_strengths, candidate_positions, actual_votes, county_positions)
        total_error = optimize.compute_total_error_for_all_counties(candidate_strengths, candidate_positions, actual_votes, county_positions)

    if config.method == "mcmc_error_min":
        print("error min")
        candidate_strengths, candidate_positions, county_positions, total_error = optimize.optimize_mcmc_error_min(actual_votes, candidate_strengths, candidate_positions, county_positions)

    if config.method == "mcmc_posterior_max":
        print("posterior max")
        candidate_strengths, candidate_positions, county_positions, total_error, step_no = optimize.optimize_mcmc_posterior_max(actual_votes, candidate_strengths, candidate_positions, county_positions, step_no, candidates, counties)

    if config.method == "scipy_minimize":
        candidate_strengths, candidate_positions, county_positions, total_error = optimize.optimize_minimize(actual_votes, candidate_strengths, candidate_positions, county_positions)

    if config.method == "scipy_basinhopping":
        candidate_strengths, candidate_positions, county_positions, total_error = optimize.optimize_basinhopping(actual_votes, candidate_strengths, candidate_positions, county_positions)

    # write to solution file unless --no-write-solution is specified
    if config.write_solution:
        solution_io.write_solution(config.solution_file, candidate_positions, candidate_strengths, county_positions, total_error, step_no)

    # print candidates
    for candidate, strength, position in sorted(zip(candidates, candidate_strengths, candidate_positions), key=lambda x: -x[1]):
        print("%-11s" % (candidate), strength, position)

    # visualization logic below
    counties_of_interest = list(range(len(county_positions)))

    if config.num_counties_to_plot:
        counties_of_interest = list(range(config.num_counties_to_plot))

    if config.plot_individual_counties:
        visualize.plot_individual_counties(
            candidates,
            counties,
            candidate_strengths,
            candidate_positions,
            np.array([county_positions[i] for i in counties_of_interest]),
            np.array([actual_votes[i] for i in counties_of_interest]))

    if config.plot_all_counties:
        visualize.plot_all_counties(candidates, counties, candidate_strengths, candidate_positions, county_positions, actual_votes)

    if config.plot_probabilities:
        visualize.plot_probabilities(config.solution_file)

    if config.plot_positions:
        visualize.plot_positions(config.solution_file)


if __name__ == '__main__':
    main()