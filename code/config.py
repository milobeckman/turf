
## config.py
## this file handles command line arguments

import argparse
import numpy as np

config = argparse.Namespace()

def parse_args():
    parser = argparse.ArgumentParser("Find the optimal arrangement of candidates and counties given a set of real-world data.")

    # Set random seed to none unless otherwise specified
    parser.add_argument("--set-seed", dest="seed", type=int, default=None, help="Initialize the internal state of the random number generator.")
    
    # four implementations we've tried so far (none work as desired)
    parser.add_argument("--mcmc-error-min", dest="method", action='store_const', const="mcmc_error_min", help="Use old MCMC implementation (essentially a stochastic gradiant descent).")
    parser.add_argument("--mcmc-posterior-max", dest="method", action='store_const', const="mcmc_posterior_max", help="Use the MCMC method, return the arrangement with highest posterior probability.")
    parser.add_argument("--scipy-minimize", dest="method", action='store_const', const="scipy_minimize", help="Use the scipy miminize method to optimize.")
    parser.add_argument("--scipy-basinhopping", dest="method", action='store_const', const="scipy_basinhopping", help="Use the scipy basinhopping algorithm to optimize.")
    parser.add_argument("--single-inner", dest="method", action='store_const', const="single_inner", help="Simply find the best voter positions for the candidate positions")

    # path to data file, which is a tab-separated plaintext file with county-level voting data
    parser.add_argument("--data-file", "-d", type=str, help="The path to the data file recording the actual votes.")

    # handling of solution files, which is a python-y plaintext file with current best stance/strength/position choices
    parser.add_argument("--solution-file", "-s", type=str, help="Where to read and write the best known solution.")
    parser.add_argument("--no-read-solution", dest="read_solution", action="store_false", default=True, help="Ignore the current best file ient and start with a random state.")
    parser.add_argument("--no-write-solution", dest="write_solution", action="store_false", default=True, help="Don't write a new best file.")
    parser.add_argument("--save-latest", dest="save_latest", action="store_true", default=False, help="Save the latest step to the solution file, rather than the best.")
    parser.add_argument("--use-latest", dest="use_latest", action="store_true", default=False, help="Use the most recent step from _positions, rather than the best.")
    parser.add_argument("--write-every-best-solution", action="store_true", default=False, help="Write a file every time we find a new best.")
    parser.add_argument("--write-every-step", action="store_true", default=False, help="Write a file every time we take a new step.")

    # run specifications
    parser.add_argument("--num-burn", "-b", type=int, default=0, help="How many iterations to burn before recording steps.")
    parser.add_argument("--num-iterations", "-n", type=int, default=0, help="How many iterations to run the outer loop.")
    parser.add_argument("--num-counties", "-v", type=int, help="If specified, use this many counties instead of all of them.")
    parser.add_argument("--candidate", "-c", dest="candidates", type=int, nargs='*', help="If specified, use these candidate indices instead of all of them.")

    # ignore this
    parser.add_argument("--dtype", type=str, default=">f8", help="Datatype for numpy arrays")

    # theoretically this all works for any dimension of ideology space, though we're sticking to R2 so far
    parser.add_argument("--dimensions", type=int, default=2, help="How many dimensions to make the space we're locating candidates and voters in")

    # adjust the way TURF calculates TPE
    parser.add_argument("--MAPE", dest="error_mode", action='store_const', const="MAPE", default="MSE", help="Use mean absolute percentage error to punish near-zero errors")

    # two hacky flags for improving results of scipy implementations in certain cases (see 02.implementation, S3.4)
    parser.add_argument("--fix-strength", action="store_true", default=False, help="Fix all candidate strengths to 1.")
    parser.add_argument("--fix-positions", action="store_true", default=False, help="Fix one candidate at the origin and another at y = 0.")
    parser.add_argument("--strength-scaling-factor", type=float, default=1, help="Scale the strength args by this factor into and out of the arg vec used by scipy minimizers.")

    # perturbation amounts for MCMC-based implementations
    parser.add_argument("--perturb-strength-mean", type=float, default=0, help="Doesn't actually do anything, since the strength gets normalized.") #### KILL THIS
    parser.add_argument("--perturb-strength-stddev", type=float, default=.25, help="Standard deviation of the perturbation factor for candidates' strength.")
    parser.add_argument("--perturb-position-stddev", type=float, default=.25, help="Standard deviation of the perturbation factor for candidates' position.")

    # for implementations which use strength priors, prior settings
    parser.add_argument("--prior-strength-loc", type=float, default=0, help="Loc (mean - 1) of the lognormal prior probability distribution for candidates' strength.")
    parser.add_argument("--prior-strength-stddev", type=float, default=0.25, help="Standard deviation of the lognormal prior probability distribution for candidates' strength.")
    parser.add_argument("--prior-location-stddev", type=float, default=4.0, help="Standard deviation of the bivariate prior probability distribution for candidates' postitions.")
    parser.add_argument("--prior-county-spread-stddev", type=float, default=0.25, help="Standard deviation of the bivariate prior probability distribution for the spread of counties amongst the neighborhoods of each candidate.")

    # candidate/county arrangement visualizations
    parser.add_argument("--plot-all-counties", type=str, help="File in which to plot all candidates and counties.")
    parser.add_argument("--plot-every", type=int, default=0, help="If specified, plots all candidates and counties ever n steps, where n is specified.")
    
    # test visualizations to make sure "inner loop" is working by plotting individual counties in error heat map
    parser.add_argument("--plot-individual-counties", type=str, help="File in which to plot individual counties.") # test flag to see if the "inner loop" is working
    parser.add_argument("--num-counties-to-plot", type=int, help="If specified, use this many counties in the individual plots instead of all of them.")

    # Plot trace for candidate positions
    parser.add_argument("--plot-positions", action="store_true", default=False, help="If specified, plot the path of accepted steps for each candidiate and histograms of the recovered probability distribution.")
    parser.add_argument("--plot-probabilities", action="store_true", default=False, help="If specified, plot the trace of probabilities throughout the walk.")

    parser.parse_args(namespace=config)
    
    
    # this is just the range [0,885] in a random order, used if the --num-coutnies flag is on
    config.rand_order_of_counties = np.array([848, 437, 652, 226, 223, 36, 521, 95, 821, 593, 383, 103, 258, 868, 246, 796, 427, 792, 270, 607, 548, 712, 287, 816, 767, 148, 222, 362, 149, 398, 392, 588, 682, 236, 71, 574, 687, 711, 177, 449, 88, 597, 298, 850, 228, 411, 660, 77, 61, 650, 455, 828, 230, 596, 552, 345, 53, 32, 55, 526, 227, 166, 737, 361, 211, 285, 598, 566, 503, 418, 793, 800, 141, 583, 844, 269, 847, 264, 656, 293, 414, 845, 572, 164, 807, 189, 368, 430, 232, 745, 573, 306, 471, 312, 704, 476, 87, 356, 797, 42, 581, 315, 15, 243, 591, 696, 316, 52, 219, 134, 878, 420, 493, 341, 466, 178, 444, 393, 18, 551, 86, 235, 160, 245, 443, 319, 750, 769, 529, 825, 41, 657, 51, 452, 515, 396, 94, 778, 80, 841, 677, 465, 65, 707, 547, 508, 623, 651, 397, 610, 43, 326, 589, 654, 334, 492, 110, 728, 288, 629, 781, 708, 242, 349, 388, 843, 133, 142, 163, 549, 181, 337, 747, 518, 22, 169, 538, 690, 794, 565, 409, 730, 375, 531, 720, 150, 432, 815, 710, 456, 813, 782, 167, 168, 407, 54, 73, 333, 135, 611, 527, 401, 577, 771, 453, 512, 462, 6, 374, 424, 740, 533, 84, 772, 502, 366, 885, 48, 343, 701, 458, 147, 154, 367, 842, 96, 278, 139, 203, 20, 851, 479, 234, 273, 200, 865, 791, 811, 560, 719, 303, 217, 72, 523, 128, 431, 404, 291, 563, 325, 63, 655, 83, 192, 174, 717, 302, 457, 132, 495, 615, 336, 627, 733, 875, 301, 237, 29, 830, 752, 634, 626, 664, 251, 35, 99, 519, 739, 876, 260, 764, 74, 335, 279, 622, 864, 70, 482, 406, 240, 143, 517, 463, 79, 151, 685, 403, 467, 638, 665, 706, 775, 605, 633, 27, 155, 543, 14, 882, 592, 666, 9, 275, 689, 639, 569, 684, 567, 870, 126, 858, 702, 224, 127, 11, 28, 263, 350, 137, 68, 642, 670, 534, 81, 787, 37, 109, 715, 46, 766, 470, 386, 389, 281, 186, 582, 609, 16, 198, 113, 344, 255, 863, 817, 92, 321, 556, 713, 619, 415, 327, 646, 698, 705, 612, 516, 156, 478, 405, 570, 370, 768, 872, 176, 220, 857, 207, 130, 884, 668, 369, 129, 553, 440, 445, 871, 442, 674, 114, 62, 693, 835, 678, 307, 394, 108, 100, 44, 8, 838, 347, 248, 562, 840, 694, 659, 783, 676, 238, 624, 571, 511, 182, 725, 351, 204, 284, 873, 290, 376, 530, 438, 595, 773, 741, 313, 594, 3, 429, 727, 726, 257, 66, 500, 834, 423, 483, 721, 426, 809, 618, 121, 24, 484, 451, 860, 310, 748, 586, 473, 399, 408, 136, 718, 474, 758, 50, 743, 587, 140, 722, 697, 239, 60, 709, 34, 123, 497, 104, 836, 206, 649, 213, 93, 608, 180, 45, 265, 342, 837, 196, 883, 691, 106, 412, 564, 277, 559, 814, 4, 877, 489, 648, 468, 31, 40, 215, 433, 25, 667, 854, 880, 205, 785, 661, 233, 320, 537, 422, 395, 117, 510, 331, 662, 216, 202, 384, 490, 171, 806, 21, 283, 606, 801, 647, 554, 38, 419, 10, 729, 173, 544, 536, 64, 90, 26, 616, 441, 580, 355, 630, 191, 671, 30, 381, 759, 599, 827, 542, 558, 76, 454, 578, 208, 673, 555, 267, 348, 339, 340, 541, 683, 308, 0, 481, 119, 636, 56, 688, 354, 7, 380, 252, 300, 780, 295, 522, 229, 318, 439, 600, 641, 210, 856, 268, 869, 416, 849, 812, 602, 757, 802, 776, 614, 282, 259, 637, 57, 247, 575, 111, 472, 152, 170, 358, 201, 112, 461, 131, 513, 250, 576, 632, 256, 280, 742, 789, 329, 723, 703, 643, 352, 377, 286, 417, 159, 679, 799, 294, 506, 120, 153, 545, 832, 699, 763, 535, 410, 145, 714, 357, 209, 69, 760, 829, 184, 296, 765, 425, 621, 102, 695, 724, 197, 818, 124, 631, 734, 879, 859, 324, 187, 774, 125, 826, 746, 162, 122, 428, 700, 736, 292, 364, 359, 881, 421, 47, 658, 144, 539, 525, 744, 261, 568, 716, 353, 402, 459, 532, 756, 372, 620, 276, 680, 304, 788, 686, 520, 480, 385, 450, 67, 839, 338, 546, 804, 861, 805, 172, 731, 12, 413, 528, 862, 212, 498, 762, 157, 822, 309, 179, 314, 244, 855, 165, 777, 262, 249, 501, 188, 371, 214, 97, 373, 391, 640, 866, 195, 507, 464, 194, 550, 271, 175, 382, 790, 183, 199, 379, 494, 601, 823, 17, 770, 645, 635, 753, 505, 653, 115, 477, 692, 378, 225, 363, 311, 390, 346, 732, 330, 675, 761, 272, 274, 681, 82, 499, 460, 751, 798, 755, 146, 786, 138, 193, 254, 628, 322, 810, 91, 107, 672, 754, 447, 434, 2, 5, 387, 297, 617, 13, 101, 161, 644, 852, 332, 436, 323, 365, 749, 58, 603, 116, 779, 299, 820, 846, 735, 509, 491, 469, 585, 218, 486, 584, 221, 540, 738, 49, 604, 784, 819, 485, 305, 435, 487, 504, 514, 400, 78, 874, 190, 75, 33, 241, 289, 1, 663, 795, 803, 853, 231, 19, 266, 328, 831, 808, 496, 867, 59, 625, 89, 158, 185, 23, 253, 561, 590, 317, 360, 118, 488, 557, 524, 579, 98, 39, 105, 824, 446, 448, 669, 475, 833, 613, 85])

















