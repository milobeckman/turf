
## optimize.py
## this file contains the core optimization methods (which don't work as desired)

from __future__ import print_function, division

import math
import random
from functools import reduce
from operator import mul
import numpy as np
from scipy.optimize import leastsq, minimize, basinhopping
from scipy.special import expit
from scipy.stats import lognorm
from scipy.stats.mstats import gmean

from config import config
import solution_io
import visualize

nCOUNTY_ERRORS = 0
MAX_COUNTY_ERRORS = 1000 # CHANGE THIS
MIN_ACC_RATE = 0.1
MAX_ACC_RATE = 0.6



## IMPLEMENTATIONS


# IMPLEMENTATION 1: MCMC Posterior Max (see 02.implementation, S2.1)
def optimize_mcmc_posterior_max(actual_votes, candidate_strengths=None, candidate_positions=None, county_positions=None, step_no=0, candidates=None, counties=None):
   
    num_votes = actual_votes.shape[0]
    num_candidates = actual_votes.shape[1]
    steps_taken = 0
    steps_attempted = 0
    
    # if no solution is passed in, initialize candidates to a random perturbation from (p,q),s=(0,0),1 
    if candidate_strengths is None:
        candidate_strengths = np.ones((num_candidates, ), dtype=config.dtype)
        candidate_strengths = perturb_candidate_strengths(candidate_strengths)
    if candidate_positions is None:
        candidate_positions = np.zeros((num_candidates, config.dimensions), dtype=config.dtype)
        candidate_positions = perturb_candidate_positions(candidate_positions)
    
    # if no solution is passed in, initialize counties to the origin
    if county_positions is None:
        county_positions = np.zeros((num_votes, config.dimensions), dtype=config.dtype)

    # compute the posterior probability of our current solution (see helper functions below)
    prior_vector, probability_vector = compute_posterior_probability(candidate_strengths, candidate_positions, county_positions, actual_votes)

    prior_strength, prior_candidate_location, prior_county_location, prior_spread = prior_vector
    prior, likelihood, posterior = probability_vector

    # save initial arrangement as current best
    best_candidate_positions = candidate_positions.copy()
    best_candidate_strengths = candidate_strengths.copy()
    best_county_positions = county_positions.copy()
    best_posterior = posterior

    print("Initial posterior:", posterior)

    # this will be updated to True if (a) nCOUNTY_ERRORS exceeds MAX_COUNTY_ERRORS or (b) acceptance rate is below MIN_ACC_RATE after 100+ attempted steps
    bad_parameters = False

    # iteratively perturb candidates (outer loop) and optimize counties (inner loop), accept perturbation according to metropolis-hastings
    try:
        while step_no < config.num_iterations and not bad_parameters:
            
            if nCOUNTY_ERRORS > MAX_COUNTY_ERRORS:
                bad_parameters = True
            
            if steps_attempted > 100 and (steps_taken < steps_attempted*MIN_ACC_RATE or steps_taken > steps_attempted*MAX_ACC_RATE):
                bad_parameters = True
            
        # for i in range(config.num_iterations):
            
            # step_no is distinct from i because it is preserved across multiple runs of the same optimization process
            # this is used for outputting images every m steps (the flag --plot-every)
            steps_attempted += 1
            
            # perturb candidates (outer loop)
            new_candidate_positions = perturb_candidate_positions(candidate_positions.copy())
            new_candidate_strengths = perturb_candidate_strengths(candidate_strengths.copy())
            
            # TODO: fix candidate positions under translation / rotation
            
            # optimize counties (inner loop)
            new_county_positions = find_minimal_error_positions_for_all_counties(new_candidate_strengths, new_candidate_positions, county_positions, actual_votes)
            prior_vector, probability_vector = compute_posterior_probability(new_candidate_strengths, new_candidate_positions, new_county_positions, actual_votes)
            
            prior_strength, prior_candidate_location, prior_county_location, prior_spread = prior_vector
            new_prior, new_likelihood, new_posterior = probability_vector

            # save this arrangement if it's our new best
            if new_posterior > best_posterior:

                print("New best posterior!:", new_posterior, end=" ")
                best_candidate_positions = new_candidate_positions.copy()
                best_candidate_strengths = new_candidate_strengths.copy()
                best_county_positions = new_county_positions.copy()
                best_prior = new_prior
                best_likelihood = new_likelihood
                best_posterior = new_posterior
                
                # write this arrangement to our solution file if we're writing every new best solution
                if config.write_every_best_solution:
                    solution_io.write_solution(config.solution_file + "-%09.6f" % best_total_error, best_candidate_positions, best_candidate_strengths, best_county_positions, best_total_error, step_no)
            # otherwise, just report the new posterior
            else:
                print(step_no,"New posterior:", new_posterior, end=" ")
            
            randval = random.random()
            ratio = step_probability_from_posteriors(posterior, new_posterior)
            # metropolis-hastings criterion (probabilistically accept new step)
            if randval < ratio:
                # print('\n',posterior_probability, 'vs', new_posterior_probability)
                # print(randval, ratio) 
                print("[ACCEPT STEP!]")
                step_no += 1
                steps_taken += 1
                candidate_positions = new_candidate_positions
                candidate_strengths = new_candidate_strengths
                county_positions = new_county_positions
                # prior_strength = new_prior_strength
                # prior_location = new_prior_location
                prior = new_prior
                likelihood = new_likelihood
                posterior = new_posterior

                if config.write_every_step and step_no > config.num_burn:
                    solution_io.write_positions( config.solution_file, candidate_positions, candidate_strengths, county_positions, step_no )
                    solution_io.write_probabilities( config.solution_file, prior_vector, probability_vector, step_no )

            else:
                # print('\n',posterior_probability, 'vs', new_posterior_probability)
                # print(randval, ratio) 
                print("[reject]")
                # print(randval, ratio)
          
            # make an image if the flag --plot-every is on
            if config.plot_every:
                # check that this is a plotting iteration
                if step_no % config.plot_every == 0  and step_no > config.num_burn:
                    print("Step " + str(step_no) + ", plotting.")
                    visualize.plot_all_counties(candidates, counties, candidate_strengths, candidate_positions, county_positions, actual_votes, step_no)
    
    # user can do a keyboard interrupt to cut optimization short and save current best solution
    except KeyboardInterrupt as e:
        print("Aborting optimization due to keyboard interrupt and returning current best")
        pass

    print("Accepted "+str(100*steps_taken/float(steps_attempted))[:5]+"% of steps.")

    print(steps_taken,"steps taken out of",steps_attempted,"steps attempted")
    
    # use the flag --save-latest when doing real MCMC stuff
    if config.save_latest:
        return candidate_strengths, candidate_positions, county_positions, posterior, step_no
    else:
        # this is weird because we're saving the most recent step no, and then restarting from the best (which has a diff step no)
        return best_candidate_strengths, best_candidate_positions, best_county_positions, best_posterior, step_no

# IMPLEMENTATION 2: MCMC Error Min (see 02.implementation, S2.2)
def optimize_mcmc_error_min(actual_votes, candidate_strengths=None, candidate_positions=None, county_positions=None, step_no=0, candidates=None, counties=None):
   
    num_votes = actual_votes.shape[0]
    num_candidates = actual_votes.shape[1]
    
    # if no solution is passed in, initialize candidates to a random perturbation from (p,q),s=(0,0),1 
    if candidate_strengths is None:
        candidate_strengths = np.ones((num_candidates, ), dtype=config.dtype)
        candidate_strengths = perturb_candidate_strengths(candidate_strengths)
    if candidate_positions is None:
        candidate_positions = np.zeros((num_candidates, config.dimensions), dtype=config.dtype)
        candidate_positions = perturb_candidate_positions(candidate_positions)
    
    # if no solution is passed in, initialize counties to the origin
    if county_positions is None:
        county_positions = np.zeros((num_votes, config.dimensions), dtype=config.dtype)

    # TODO: fix candidate positions under translation / rotation

    # compute the total error of our current solution (see helper functions below)
    total_error = compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)

    print("Initial error:", total_error)

    # save initial arrangement as current best
    best_candidate_positions = candidate_positions.copy()
    best_candidate_strengths = candidate_strengths.copy()
    best_county_positions = county_positions.copy()
    best_total_error = total_error


    # iteratively perturb candidates (outer loop) and optimize counties (inner loop), accept perturbation if new step is better
    try:
        for i in range(config.num_iterations):
            
            # step_no is distinct from i because it is preserved across multiple runs of the same optimization process
            # this is used for outputting images every m steps (the flag --plot-every)
            step_no += 1
            
            # perturb candidates (outer loop)
            new_candidate_positions = perturb_candidate_positions(candidate_positions.copy())
            new_candidate_strengths = perturb_candidate_strengths(candidate_strengths.copy())
            
            # TODO: fix candidate positions under translation / rotation
            
            # optimize counties (inner loop)
            new_county_positions = find_minimal_error_positions_for_all_counties(new_candidate_strengths, new_candidate_positions, county_positions, actual_votes)
            new_total_error = compute_total_error_for_all_counties(new_candidate_strengths, new_candidate_positions, new_county_positions, actual_votes)
            
            # save this arrangement if it's our new best
            if new_total_error < best_total_error:
                print("New best error!:", new_total_error, end=" ")
                best_candidate_positions = new_candidate_positions.copy()
                best_candidate_strengths = new_candidate_strengths.copy()
                best_county_positions = new_county_positions.copy()
                best_total_error = new_total_error
                
                # write this arrangement to our solution file if we're writing every new best solution
                if config.write_every_best_solution:
                    solution_io.write_solution(config.solution_file + "-%09.6f" % best_total_error, best_candidate_positions, best_candidate_strengths, best_county_positions, best_total_error, step_no)
            
            # otherwise, just report the new error
            else:
                print("New error:", new_total_error, end=" ")
            
            # deterministically accept new step if it's better (this is REALLY not MCMC...)
            if random.random() < step_probability_from_errors(total_error, new_total_error):
                print("[ACCEPT STEP!]")
                candidate_positions = new_candidate_positions
                candidate_strengths = new_candidate_strengths
                county_positions = new_county_positions
                total_error = new_total_error
            else:
                print("[reject]")
          
            # make an image if the flag --plot-every is on
            if config.plot_every:
                # check that this is a plotting iteration
                if step_no % config.plot_every == 0:
                    print("Step " + str(step_no) + ", plotting current best.")
                    visualize.plot_all_counties(candidates, counties, candidate_strengths, candidate_positions, county_positions, actual_votes, step_no)
    
    # user can do a keyboard interrupt to cut optimization short and save current best solution
    except KeyboardInterrupt as e:
        print("Aborting optimization due to keyboard interrupt and returning current best")
        pass

    return best_candidate_strengths, best_candidate_positions, best_county_positions, best_total_error, step_no


# IMPLEMENTATION 3: Scipy Minimize (see 02.implementation, S2.3)
def optimize_minimize(actual_votes, candidate_strengths=None, candidate_positions=None, county_positions=None):
    num_votes = actual_votes.shape[0]
    num_candidates = actual_votes.shape[1]
    
    # if no solution is passed in, initialize candidates to a random perturbation from (p,q),s=(0,0),1 
    if candidate_strengths is None:
        candidate_strengths = np.ones((num_candidates, ), dtype=config.dtype)
        candidate_strengths = perturb_candidate_strengths(candidate_strengths)
    if candidate_positions is None:
        candidate_positions = np.zeros((num_candidates, config.dimensions), dtype=config.dtype)
        candidate_positions = perturb_candidate_positions(candidate_positions)
    
    # if no solution is passed in, initialize counties to the origin
    if county_positions is None:
        county_positions = np.zeros((num_votes, config.dimensions), dtype=config.dtype)

    # initialize the arg vector and county position cache for optimization
    initial_arg_vec = candidate_strengths_and_positions_to_arg_vec(candidate_strengths, candidate_positions, num_candidates)
    init_cached_initial_county_position(county_positions)

    # use scipy minimize to find an optimal candidate arrangement, assuming optimal county arrangement for a given candidate arrangement
    result = minimize(
        total_error_for_minimal_error_configuration_of_counties_for_candidates_with_cached_initial_county_position,
        initial_arg_vec,
        (num_candidates, actual_votes))
    
    print(result)
    candidate_strengths, candidate_positions = arg_vec_to_candidate_strengths_and_positions(result.x, num_candidates)
    county_positions = find_minimal_error_positions_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
    error = compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
    return candidate_strengths, candidate_positions, county_positions, error


# IMPLEMENTATION 4: Scipy Basin-Hopping (see 02.implementation, S2.4)
def optimize_basinhopping(actual_votes, candidate_strengths=None, candidate_positions=None, county_positions=None):
    num_votes = actual_votes.shape[0]
    num_candidates = actual_votes.shape[1]
    
    # if no solution is passed in, initialize candidates to a random perturbation from (p,q),s=(0,0),1 
    if candidate_strengths is None:
        candidate_strengths = np.ones((num_candidates, ), dtype=config.dtype)
        candidate_strengths = perturb_candidate_strengths(candidate_strengths)
    if candidate_positions is None:
        candidate_positions = np.zeros((num_candidates, config.dimensions), dtype=config.dtype)
        candidate_positions = perturb_candidate_positions(candidate_positions)
     
    # if no solution is passed in, initialize counties to the origin
    if county_positions is None:
        county_positions = np.zeros((num_votes, config.dimensions), dtype=config.dtype)

    # initialize the arg vector and county position cache for optimization
    initial_arg_vec = candidate_strengths_and_positions_to_arg_vec(candidate_strengths, candidate_positions, num_candidates)
    init_cached_initial_county_position(county_positions)

    # create the callback function passed into scipy basinhopping
    log_local_minimum = produce_log_local_minimum_lambda(num_candidates, actual_votes, county_positions)
    
    # use scipy basinhopping to find an optimal candidate arrangement, assuming optimal county arrangement for a given candidate arrangement
    result = basinhopping(
        total_error_for_minimal_error_configuration_of_counties_for_candidates_with_cached_initial_county_position,
        initial_arg_vec,
        niter=config.num_iterations,
        T=2.0,
        stepsize=5.0,
        minimizer_kwargs={
            "args": (num_candidates, actual_votes)
        },
        callback=log_local_minimum,
        disp=True
    )
    
    print(result)
    candidate_strengths, candidate_positions = arg_vec_to_candidate_strengths_and_positions(result.x, num_candidates)
    county_positions = find_minimal_error_positions_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
    error = compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
    return candidate_strengths, candidate_positions, county_positions, error


## GENERAL TURF HELPER FUNCTIONS

# computes distance
def distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

# computes salience, s/1+d
def salience_for_candidate_and_county(candidate_strength, candidate_position, county_position):
    return candidate_strength / (1 + distance(candidate_position, county_position))

# predicts vote breakdown in a county (proportional to salience, sum to 1)
def predict_vote_for_single_county(candidate_strengths, candidate_positions, county_position):
    saliences = [salience_for_candidate_and_county(candidate_strength, candidate_position, county_position) for candidate_strength, candidate_position in zip(candidate_strengths, candidate_positions)]
    total_salience = sum(saliences)
    return np.array([salience / total_salience for salience in saliences], dtype=config.dtype)

# computes (actual - expected) for a single (candidate,county) pair
def compute_error_components_for_single_county(candidate_strengths, candidate_positions, county_position, actual_vote):
    predicted_vote = predict_vote_for_single_county(candidate_strengths, candidate_positions, county_position)
    return predicted_vote - actual_vote

# scipy leastsq needs the parameter over which to optimize as the first arg -- this is hacky but I want to maintain the same arg order elsewhere
def compute_error_components_for_single_county_leastsq_helper_version(county_position, candidate_strengths, candidate_positions, actual_vote):
    return compute_error_components_for_single_county(candidate_strengths, candidate_positions, county_position, actual_vote)

# computes MSE for all candidates for a single county (this is "county prediction error" or TPE)
def compute_total_error_for_single_county(candidate_strengths, candidate_positions, county_position, actual_vote):
    
    error_mode = config.error_mode
    
    # straightforward mean squared error
    if error_mode == "MSE":
        errors = compute_error_components_for_single_county(candidate_strengths, candidate_positions, county_position, actual_vote)
        return sum(errors * errors) / len(errors)
    
    # mean absolute percentage error, punishes 0.0 when correct answer is 0.1 more than 0.5 when correct answer is 0.6
    if error_mode == "MAPE":
        predicted_vote = predict_vote_for_single_county(candidate_strengths, candidate_positions, county_position)
        abs_pct_errors = [abs((predicted_vote[i] - actual_vote[i]) / actual_vote[i]) for i in range(len(actual_vote))]
        return sum(abs_pct_errors)

# computes total MSE for all counties and candidates (this is "electorate prediction error" or EPE; this is what we want to minimize)
def compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes):
    return sum(
        compute_total_error_for_single_county(candidate_strengths, candidate_positions, county_position, actual_vote)
        for county_position, actual_vote
        in zip(county_positions, actual_votes)
    )


## PRIOR CALCULATIONS

# uses a log-normal distribution centered at 1 with stddev specified by user
def prior_strength_log_normal(candidate_strengths, candidate_positions, county_positions, actual_votes):
    return math.exp(-np.sum(np.log(candidate_strengths) ** 2 / ( 2 * config.prior_strength_stddev ** 2)))

# uses a bivariate gaussian on the absolute locations of each x,y coordinate
def prior_location_bivariate_gaussian_absolute(candidate_strengths, candidate_positions, county_positions, actual_votes):
    distances = np.sqrt(np.sum(candidate_positions ** 2, axis=1))
    return math.exp(-np.sum(distances / (2 * config.prior_location_stddev ** 2)))

# uses a bivariate guassian on the relative location of each x,y coordinate to the corresponding x,y mean (not Eucl. dist. from centroid)
# this is the same as above, if you use flag --fix-positions (centroid = 0,0)
def prior_location_bivariate_gaussian_relative(candidate_strengths, candidate_positions, county_positions, actual_votes):
    centroid = np.mean(candidate_positions, axis=0)
    candidate_positions_relative = candidate_positions - centroid
    distances = np.sqrt(np.sum(candidate_positions_relative ** 2, axis=1))
    return math.exp(-np.mean(distances / (2 * config.prior_location_stddev ** 2)))

def prior_county_location_bivariate_gaussian(candidate_strengths, candidate_positions, county_positions, actual_votes):
    distances = np.sqrt(np.sum(county_positions ** 2, axis=1))
    return math.exp(-np.mean(distances / (2 * config.prior_location_stddev ** 2)))

# checks how many counties are nearest each candidate, "should" be 1/n each
def prior_county_spread(candidate_strengths, candidate_positions, county_positions, actual_votes):
    
    # count of how many counties are nearest candidate i
    neighborhood_populations = [0]*len(candidate_positions)
    
    for county_position in county_positions:
        distances = [distance(county_position, candidate_position) for candidate_position in candidate_positions]
        nearest = distances.index(min(distances))
        neighborhood_populations[nearest] += 1
    
    
    # print(neighborhood_populations)
    
    # right now i'm just using a bivariate gaussian on a pretty arbitrary manipulation of this value -- this is probably dumb
    
    # "deviations" is how different the true county spread is from [1/n, 1/n, ..., 1/n]
    deviations = np.array([abs(x/len(county_positions) - 1/len(candidate_positions)) for x in neighborhood_populations])
    prior = math.exp(-np.sum(deviations**2 / (2 * config.prior_county_spread_stddev ** 2)))
    
    return prior




## INNER LOOP HELPER FUNCTIONS

# place a single county at the min-error/max-posterior position (they are equivalent since we have no prior on position) for a given arrangement of candidates
def find_minimal_error_position_for_single_county(candidate_strengths, candidate_positions, old_county_position, actual_vote):
    args = (candidate_strengths, candidate_positions, actual_vote)
    
    # use scipy function leastsq to minimize error
    result = leastsq(compute_error_components_for_single_county_leastsq_helper_version, old_county_position, args=args, full_output=True)
    if result[4] not in (1, 2, 3, 4):
        global nCOUNTY_ERRORS
        nCOUNTY_ERRORS += 1
        print("Couldn't find county position:", result[3])
    return result[0]

# run inner loop on each county passed in
def find_minimal_error_positions_for_all_counties(candidate_strengths, candidate_positions, old_county_positions, actual_votes):
    county_positions = np.array([
        find_minimal_error_position_for_single_county(candidate_strengths, candidate_positions, old_county_position, actual_vote)
        for actual_vote, old_county_position
        in zip(actual_votes, old_county_positions)
    ], dtype=config.dtype)
    return county_positions



## HELPER FUNCTIONS FOR MCMC-POSTERIOR-MAX AND MCMC-ERROR-MIN

# perturb candidate strengths according to the step sizes passed in, normalize to geometric mean=1
def perturb_candidate_strengths(candidate_strengths):
    num_candidates = candidate_strengths.shape[0]
    candidate_strengths *= np.random.lognormal(config.perturb_strength_mean, config.perturb_strength_stddev, size=num_candidates)
    
    # we normalize to make strength priors reasonable (e.g. four candidates with strengths 1,4,4,4 should have same prior as 0.25,1,1,1)
    scaling_factor = gmean(list(candidate_strengths))
    candidate_strengths /= scaling_factor

    return candidate_strengths

# perturb candidate positions according to the step sizes passed in
def perturb_candidate_positions(candidate_positions):
    num_candidates = candidate_positions.shape[0]
    candidate_positions += np.random.normal(0, config.perturb_position_stddev, size=(num_candidates, config.dimensions))

    # fix canddiates under translation and rotation
    if config.fix_positions:
        candidate_positions = fix_under_translation(candidate_positions)
        candidate_positions = fix_under_rotation(candidate_positions)

    return candidate_positions

# translate candidate positions so centroid = (0,0)
def fix_under_translation(candidate_positions):
    centroid = np.mean(candidate_positions, axis=0)
    return candidate_positions - centroid

# rotate candidate positions so K1K2 || x-axis
def fix_under_rotation(candidate_positions, K1=0, K2=1, x=0, y=1):
    K1x,K1y = candidate_positions[K1][x],candidate_positions[K1][y]
    K2x,K2y = candidate_positions[K2][x],candidate_positions[K2][y]
    angle_with_xaxis = np.arctan2(K2y-K1y,K2x-K1x)
    cos,sin = np.cos(-angle_with_xaxis),np.sin(-angle_with_xaxis)
    rotation_matrix = np.matrix([[cos,-sin],[sin,cos]])
    
    # rotate in x,y-plane so that K1K2 segment is parallel to x axis
    candidate_positions = np.array([rotation_matrix.dot(old_pos).tolist()[0] for old_pos in candidate_positions])
    
    return candidate_positions

# calculate the posterior probability of an arrangement (e^-err * prod[strength_priors])
def compute_posterior_probability(candidate_strengths, candidate_positions, county_positions, actual_votes):
    
    # likelihood of attaining this actual_votes data given this arrangement of candidates and counties (e^-err)
    total_error = compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
    # likelihood = math.exp(-total_error)
    # likelihood = math.exp(-total_error/2)
    likelihood = math.exp(-total_error/(2 * config.perturb_position_stddev ** 2))

    # Lognormal prior on strength, centered at 1.0
    prior_strength = prior_strength_log_normal(candidate_strengths, candidate_positions, county_positions, actual_votes)

    # Bivariate gaussian prior on location, centered at (0,0)
    prior_candidate_location = prior_location_bivariate_gaussian_absolute(candidate_strengths, candidate_positions, county_positions, actual_votes)
    
    # Bivariate guassian prior on county location, centered at (0,0)
    prior_county_location = prior_county_location_bivariate_gaussian(candidate_strengths, candidate_positions, county_positions, actual_votes)
    
    # [NEEDS WORK] Prior on county spread
    prior_spread = prior_county_spread(candidate_strengths, candidate_positions, county_positions, actual_votes)
    
    # Compute prior (put alpha + beta here?)
    # prior = prior_strength * prior_candidate_location * prior_county_location * prior_spread

    prior = ( prior_strength * prior_candidate_location * prior_county_location * prior_spread ) ** (1/4)

    # Posterior probability
    posterior = prior * likelihood

    prior_vector = [ prior_strength, prior_candidate_location, prior_county_location, prior_spread ]

    probability_vector = [ prior, likelihood, posterior ]

    return prior_vector, probability_vector

# stochastically decides whether to take next step using metropolis-hastings criterion
def step_probability_from_posteriors(old_posterior, new_posterior):
    if new_posterior > old_posterior:
        return 1
    else:
        return new_posterior / old_posterior

# deterministically decides whether to take next step based on whether it's better (this is REALLY not MCMC...)
def step_probability_from_errors(old_error, new_error):
  if new_error < old_error:
    return 1
  else:
    return 0

## HELPER FUNCTIONS FOR SCIPY-MINIMIZE AND SCIPY-BASINHOPPING

# this is the function that scipy minimize is minimizing over
def total_error_for_minimal_error_configuration_of_counties_for_candidates_with_cached_initial_county_position(arg_vec, num_candidates, actual_votes):
    # cache the county position found in the last call as our initial guess for this call (saves on runtime)
    global cached_county_position
    candidate_strengths, candidate_positions = arg_vec_to_candidate_strengths_and_positions(arg_vec, num_candidates)
    
    # given this arrangement of candidates, find optimal county placement and calculate error
    county_positions = find_minimal_error_positions_for_all_counties(candidate_strengths, candidate_positions, cached_county_position, actual_votes)
    error = compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
    cached_county_position = county_positions
    
    # report and return error
    print("error", error)
    return error

# convert from our format to scipy's format
def candidate_strengths_and_positions_to_arg_vec(candidate_strengths, candidate_positions, num_candidates):
    if config.fix_strength:
        return candidate_positions.reshape(num_candidates * config.dimensions)
    else:
        return np.concatenate((candidate_strengths, candidate_positions.reshape(num_candidates * config.dimensions)))

# convert from scipy's format to our format
def arg_vec_to_candidate_strengths_and_positions(arg_vec, num_candidates):
    if config.fix_strength:
        candidate_strengths = np.ones((num_candidates,))
        candidate_positions = arg_vec[:].reshape(num_candidates, config.dimensions)
    else:
        candidate_strengths = arg_vec[:num_candidates]
        candidate_positions = arg_vec[num_candidates:].reshape(num_candidates, config.dimensions)
    return candidate_strengths, candidate_positions

# we cache the county position found in the prev call as our initial guess for next call (saves on runtime)
def init_cached_initial_county_position(county_position):
    global cached_county_position
    cached_county_position = county_position

# create the callback function for scipy basinhopping
def produce_log_local_minimum_lambda(num_candidates, actual_votes, old_county_positions):
    def log_local_minimum(arg_vec, error, accepted):
        print("Got callback for local minimum!")
        
        candidate_strengths, candidate_positions = arg_vec_to_candidate_strengths_and_positions(arg_vec, num_candidates)
        
        county_positions = find_minimal_error_positions_for_all_counties(candidate_strengths, candidate_positions, old_county_positions, actual_votes)
        computed_error = compute_total_error_for_all_counties(candidate_strengths, candidate_positions, county_positions, actual_votes)
        
        solution_io.write_solution("basinhopping-%09.6f.json" % computed_error, candidate_positions, candidate_strengths, county_positions, computed_error)

