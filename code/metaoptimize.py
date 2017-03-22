import subprocess, shlex
import time


def main():
    
    data_file = "data/test_case_2.txt"
    method = "mcmc-posterior-max"
    num_iter = 10000
    
    # an identifier for this particular run (e.g. 00120) will be inserted in place of %I (e.g. test_case_2_v00120.txt)
    solution_file = "solutions/test_case_2_v%I.txt"
    viz_file = "images/test_case_2_viz_v%I.png"
    
    # the hyperparameter choices to grid search through
    hyperparameters = ["perturb-strength-stddev","perturb-position-stddev","prior-strength-stddev","prior-location-stddev","prior-county-spread-stddev"]
    choices = dict.fromkeys(hyperparameters)
    
    choices["perturb-strength-stddev"] = [0.1,0.5,1.0]
    choices["perturb-position-stddev"] = [0.25,1.0,5.0]
    choices["prior-strength-stddev"] = [0.1,0.25,0.5]
    choices["prior-location-stddev"] = [1.0,2.0,5.0]
    choices["prior-county-spread-stddev"] = [0.025,0.25,2.5]
    
    # prepare the recursive call to grid_search
    run_command_so_far = "python main.py -d " + data_file + " -s " + solution_file + " -n " + str(num_iter) + " --plot-all-counties " + viz_file + " --" + method
    identifier_so_far = ""
    
    # call it
    grid_search(hyperparameters, choices, run_command_so_far, identifier_so_far)
    

# recursive function -- for each hyperparameter, calls 3 instances of grid_search (one for each h.p. choice) on the remaining hyperparameters
def grid_search(hyperparameters, choices, run_command_so_far, identifier_so_far):
    
    # base case: sub in the identifier and run
    if len(hyperparameters) == 0:
        args = shlex.split(run_command_so_far.replace("%I", identifier_so_far))
        #print(args)
        
        try:
            p = subprocess.Popen(args)
        except KeyboardInterrupt:
            # this doesn't do anything
            print("INTERRUPTED")
        
        return
    
    # choose the hyperparameter to iterate over
    my_hp = hyperparameters[0]
    identifier_index = 0
    run_command_so_far += " --" + my_hp + " "
    
    # call grid_search for each choice of hyperparameter
    for choice in choices[my_hp]:
        grid_search(hyperparameters[1:], choices, run_command_so_far + str(choice), identifier_so_far + str(identifier_index))
        identifier_index += 1


if __name__ == '__main__':
    main()
