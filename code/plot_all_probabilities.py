import visualize


## OUTPUT MUST BE IN SUBDIR data/metaoptimize-results/

def main():
    files_to_plot = ["test_case_2_v"]
    
    for i in range(5):
        new_files_to_plot = []
        for filename in files_to_plot:
            for i in range(3):
                new_files_to_plot += [filename + str(i)]
        files_to_plot = new_files_to_plot
    
    files_to_plot = ["data/metaoptimize-results/" + x + ".txt" for x in files_to_plot]
    
    for filename in files_to_plot:
        visualize.plot_probabilities(filename)
    

if __name__ == '__main__':
    main()