
## visualize.py
## this file generates an image for a given arrangement of candidates and counties


import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import matplotlib.cm as cm
import os

from config import config
import optimize



## I haven't really touched this code since Felix wrote it...

def plot_individual_counties(candidates, counties, candidate_strengths, candidate_positions, county_positions, actual_votes):
    fig_wh = math.ceil(len(county_positions) ** 0.5)

    fig = plt.figure(figsize=(2 * fig_wh, 2 * fig_wh))

    for fig_num, (county_position, actual_vote) in enumerate(zip(county_positions, actual_votes)):
        
        all_x_coords = [county_position[0]] + [cp[0] for cp in candidate_positions]
        all_y_coords = [county_position[1]] + [cp[1] for cp in candidate_positions]
        minx = min(all_x_coords)
        maxx = max(all_x_coords)
        miny = min(all_y_coords)
        maxy = max(all_y_coords)
        
        padx = (maxx - minx) * .33
        pady = (maxy - miny) * .33
        
        minx -= padx
        maxx += padx
        miny -= pady
        maxy += pady
        
        ax = fig.add_subplot(fig_wh, fig_wh, fig_num + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_autoscale_on(False)
        ax.axis([minx, maxx, miny, maxy])
        
        func = lambda x, y: optimize.compute_total_error_for_single_county(np.array((x, y)), candidate_strengths, candidate_positions, actual_vote)
        
        n = 64
        x = np.linspace(minx, maxx, n)
        y = np.linspace(miny, maxy, n)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = func(X[i,j], Y[i,j])
        
        ax.pcolor(X,Y,Z)
        
        for candidate, candidate_strength, candidate_position in zip(candidates, candidate_strengths, candidate_positions):
            ax.plot(candidate_position[0], candidate_position[1], 'r+', markersize=max(1, candidate_strength / 2))
            ax.text(candidate_position[0], candidate_position[1], candidate, color='w', size=6, alpha=1)
        
        ax.plot(county_position[0], county_position[1], 'g+')

    fig.savefig(config.plot_individual_counties, dpi=300, bbox_inches='tight')
    # plt.show(fig)

def plot_all_counties(candidates, counties, candidate_strengths, candidate_positions, county_positions, actual_votes, step_no=-1):

    savefile = config.plot_all_counties

    # adjust savefile name if we're plotting regularly
    if step_no > -1:
        if '/' not in savefile:
            savefile = str(step_no) + "-" + savefile
        else:
            i = savefile.rfind("/")+1
            savefile = savefile[:i] + str(step_no) + "-" + savefile[i:]

    if os.path.exists(savefile): return

    fig = plt.figure(figsize=(10,10))

    all_x_coords = [cp[0] for cp in candidate_positions]
    all_y_coords = [cp[1] for cp in candidate_positions]
    minx = min(all_x_coords)
    maxx = max(all_x_coords)
    miny = min(all_y_coords)
    maxy = max(all_y_coords)

    padx = (maxx - minx) * .33
    pady = (maxy - miny) * .33

    minx -= padx
    maxx += padx
    miny -= pady
    maxy += pady

    ax = fig.add_subplot(1, 1, 1)
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    ax.set_autoscale_on(False)
    ax.axis([minx, maxx, miny, maxy])
    ax.set_axis_bgcolor('k')

    for county, county_position, actual_vote in zip(counties, county_positions, actual_votes):
        color = 'g'
        if county[0] == "Alabama":
            color = '#E30E0E'
        elif county[0] == "Arkansas":
            color = '#AD3A32'
        elif county[0] == "Georgia":
            color = '#1F38DE'
        elif county[0] == "Massachusetts":
            color = '#A2DB65'
        elif county[0] == "Oklahoma":
            color = '#C4C4C4'
        elif county[0] == "Tennessee":
            color = '#756228'
        elif county[0] == "Texas":
            color = '#DEC823'
        elif county[0] == "Vermont":
            color = '#49DE23'
        elif county[0] == "Virginia":
            color = '#DE9623'
        ax.plot(county_position[0], county_position[1], '.', color=color, markersize=5)
 
    for candidate, candidate_strength, candidate_position in zip(candidates, candidate_strengths, candidate_positions):
        ax.plot(candidate_position[0], candidate_position[1], 'r.', markersize=10*max(0.5, candidate_strength))
        ax.text(candidate_position[0], candidate_position[1], candidate, color='w', size=6, alpha=1)
  
    fig.savefig(savefile, dpi=300, bbox_inches='tight')


def plot_probabilities(filename):

    filename = filename.replace('.txt','_probabilities.txt')
    figname = filename.replace('.txt','_probabilities.pdf')
    df = pd.read_table(filename)

    fig = plt.figure(figsize=(25,10))

    n = 0
    ix = {'Posteriors':[3,4,5],'Priors':[1,2,3]}
    for grp in ix.keys():
        n += 1
        ax = fig.add_subplot(2,1,n)
        for prob in df.columns.values[ix[grp]]:
            plt.plot(df['step_no'],df[prob],label=prob)

        plt.title(grp,fontsize=20)
        plt.legend()
        plt.ylabel(r'$P$',fontsize=30,labelpad=25,rotation=0)
        plt.tick_params(labelsize=20)

    fig.savefig(figname, transparent=True, dpi=300, bbox_inches='tight')


def plot_positions(filename,bin_width=100):

    filename = filename.replace('.txt','_positions.txt')
    figname = filename.replace('.txt','_positions.pdf')

    df = pd.read_table(filename)
    candidates = list(set([ x.split('_')[0] for x in df.columns.values if x[0] != 'T' ]))

    xmin = df[[ K+'_x' for K in candidates ]].min().min()*1.1
    xmax = df[[ K+'_x' for K in candidates ]].max().max()*1.1
    ymin = df[[ K+'_y' for K in candidates ]].min().min()*1.1
    ymax = df[[ K+'_y' for K in candidates ]].max().max()*1.1

    fig = plt.figure(figsize=(20,11*len(candidates)))
    gs = gridspec.GridSpec(len(candidates), 2)
    gs.update(wspace=0.15,hspace=0.3)

    for i in range(len(candidates)):

        K = candidates[i]
        x = df[K+'_x']
        y = df[K+'_y']
        s = df[K+'_s']
        
        ## LEFT PANEL
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 2,
                                               subplot_spec=gs[2*i],
                                               width_ratios=[3,1],wspace=0.1,
                                               height_ratios=[1,3],hspace=0.1
                                              )
        ## Subplots
        ax1 = plt.subplot(gs0[0]) # X distribution
        ax2 = plt.subplot(gs0[1]) # Title
        ax3 = plt.subplot(gs0[2]) # MCMC
        ax4 = plt.subplot(gs0[3]) # Y distribution  
        
        ## X histogram
        n, bins, patches = ax1.hist(x,bin_width,
                                     histtype='step',
                                     color='black')
        ax1.axis([xmin,xmax,0.0,1.1*max(n)])
        ax1.plot([np.mean(x),np.mean(x)],[0.0,1.1*max(n)],'r-',linewidth=5.0,alpha=0.5)
        ax1.plot([np.median(x),np.median(x)],[0.0,1.1*max(n)],'b-',linewidth=5.0,alpha=0.5)
        ax1.tick_params(labelleft='off',labelbottom='off')
        
        ## Title
        ax2.axis([0.0,1.0,0.0,1.0])
        ax2.annotate(K, xy=(0,0.75), fontsize=30, rotation=45)
        ax2.axis('off')
        
        ## MCMC
        ax3.plot(df[K+'_x'],df[K+'_y'],
                 linewidth=0.5,
                 alpha=0.5,
                 color='black')
        ax3.axis([xmin,xmax,ymin,ymax])
        ax3.tick_params(labelsize=25)
        ax3.set_xlabel(r'$X$',fontsize=40)
        ax3.set_ylabel(r'$Y$',fontsize=40,rotation=0,labelpad=20)
        
        ## Y distribution
        n, bins, patches = ax4.hist(y,bin_width,
                                     orientation='horizontal',
                                     histtype='step',
                                     color='black')
        ax4.axis([0.0,1.1*max(n),ymin,ymax])
        ax4.plot([0.0,1.1*max(n)],[np.mean(y),np.mean(y)],'r-',linewidth=5.0,alpha=0.5)
        ax4.plot([0.0,1.1*max(n)],[np.median(y),np.median(y)],'b-',linewidth=5.0,alpha=0.5)
        ax4.tick_params(labelleft='off',labelbottom='off')
            
        ## RIGHT PANEL
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2*i+1])
        
        ## Strength distribution
        ax5 = plt.subplot(gs1[0])
        n, bins, patches = ax5.hist(s,bin_width,
                                    histtype='step',
                                   color='black')
        ax5.axis([0.0,3.0,0.0,1.1*max(n)])
        ax5.plot([np.mean(s),np.mean(s)],[0.0,1.1*max(n)],'r-',linewidth=5.0,alpha=0.5)
        ax5.plot([np.median(s),np.median(s)],[0.0,1.1*max(n)],'b-',linewidth=5.0,alpha=0.5)
        ax5.tick_params(labelleft='off',labelsize=20)
        ax5.set_xlabel(r'$S$',fontsize=40)

    fig.savefig(figname, transparent=True, dpi=300, bbox_inches='tight')


