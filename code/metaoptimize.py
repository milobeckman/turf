import subprocess, shlex, os


def main():
    
    data_file = "data/test_case_2.txt"
    method = "mcmc-posterior-max"
    num_iter = 10000
    seed_no = 1
    
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
    
    # prepare the recursive call to list_of_run_commands and generate list
    run_command_so_far = ' '.join(['python main.py',
                                    '-d',data_file,
                                    '-s',solution_file,
                                    '-n',str(num_iter),
                                    '--plot-all-counties',viz_file,
                                    '--'+method,
                                    '--save-latest',
                                    '--set-seed',str(seed_no),
                                    '--write-every-step',
                                    '--plot-probabilities',
                                    '--plot-positions'])
    identifier_so_far = ""
    command_list = list_of_run_commands(hyperparameters, choices, run_command_so_far, identifier_so_far)
    
    # run commands in parallel
    run_in_parallel(command_list, 5)
  


# recursively generates a list of all combinations of hyperparameters
def list_of_run_commands(hyperparameters, choices, run_command_so_far, identifier_so_far):
    
    # base case: sub in the identifier and run
    if len(hyperparameters) == 0:
        run_command = run_command_so_far.replace("%I", identifier_so_far)
        return [run_command]
    
    # choose the hyperparameter to iterate over
    my_hp = hyperparameters[0]
    identifier_index = 0
    run_command_so_far += " --" + my_hp + " "
    
    command_list = []
    
    # call grid_search for each choice of hyperparameter
    for choice in choices[my_hp]:
        command_list += list_of_run_commands(hyperparameters[1:], choices, run_command_so_far + str(choice), identifier_so_far + str(identifier_index))
        identifier_index += 1

    return command_list


# run the list of commands in parallel, no more than max_processes at a time
def run_in_parallel(command_list, max_processes):
    
    processes = set()
    
    for command in command_list:
        processes.add(subprocess.Popen(shlex.split(command)))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])
    


if __name__ == '__main__':
    main()
