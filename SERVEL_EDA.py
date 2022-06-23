import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import os

# define location of the file
file_loc = input()
assert type(file_loc) == str

# import file
os.chdir(file_loc)
file1 = pd.ExcelFile('Resultados_Mesa_Presidenciales_1v_2021-1.xlsx')
file2 = pd.ExcelFile('Resultados_mesa_presidencial_TRICEL_2v_2021-1.xlsx')

# turn Excel sheets into DataFrames
df1_int = file1.parse('CHILE')
df1_ext = file1.parse('EXTRANJERO')
df1_tot = file1.parse('TOTAL')
df2_int= file2.parse('CHILE')
df2_ext = file2.parse('EXTRANJERO')
df2_tot = file2.parse('TOTAL')


# define function to count total number of ballots
def compute_total_ballots(col, vot_opt):
    '''
    Compute the total of votes expressed in a polling place (pp)

    :param col: numpy array of number of votes expressed in each pp
    :param vot_opt: int number of voting options
    :return: array of the same length than the column, with number of expressed votes per pp
    '''

    # check that the df1_int is a numpy array composed of a constant number of polling places
    assert type(col) == np.ndarray
    assert len(col) % vot_opt == 0

    # create empty array to be filled and returned as the function's ouput
    ballots_array = np.empty(len(col), dtype='int64')

    # iterate over the rows to compute the total of ballots per voting place
    for pp_idx in range(0,len(col),vot_opt):
        pp_rows = col[pp_idx:pp_idx+vot_opt]
        pp_total_ballots = np.sum(pp_rows)
        ballots_array[pp_idx:pp_idx+vot_opt] = pp_total_ballots

    return ballots_array


# define function to return all scores of a candidate as a numpy array
def add_int_ext(cand_name, df_int, df_ext):
    '''
    join national vote data to foreign vote data
        :param cand_name: name of the candidate, as a string
        :param df_int: ballots DataFrame, expected to be national votes
        :param df_ext: ballots DataFrame, expected to be foreign votes
        :return: all the data as a single array
        '''

    # the name of the candidate has to be uppercase
    cand_name = cand_name.upper()

    # vectorize df's col with numpy
    ar_int = df_int.loc[df_int['Candidato']==cand_name]['Porcentaje Mesa'].to_numpy()
    ar_ext = df_ext.loc[df_ext['Candidato']==cand_name]['Porcentaje Mesa'].to_numpy()

    # join the two arrays and return the result
    f_ar = np.append(ar_int, ar_ext)
    return f_ar * 100


# realize the following operations on each df:
    # clean useless entries and features, fill null values, convert in appropriate datatypes
    # drop non-expressed votes
    # sort in order to have ordered rows corresponding to the polling places
    # compute % of votes of each candidate in each polling place
for df in [df1_int, df1_ext, df2_int, df2_ext]:

    # clean the useless empty last column
    # df2_int already has the expected shape and does not need to be cleaned
    if df.columns[-1] != 'Votos TRICEL':
        df.drop(df.columns[-1], axis=1, inplace=True)
        assert df.columns[-1] == 'Votos TRICEL'
    print('1- Drop last column if not TRICEL complete')

    # clean empty rows at the bottom of the df
    df.drop(df.index[-2:], inplace=True)
    print('2 - Drop last two rows complete')

    # replace Nan values of the Votos TRICEL column by zero
    df['Votos TRICEL'].fillna(0, inplace=True)
    print('3 - Fill TRICEL complete')

    # determine if its a national or foreign ballots df
    # and the convert columns to appropriate datatypes
    print(df.dtypes)
    if df.columns[0] == 'Nro. Región':
        # convert float columns to int type
        for col in [c for c in df.columns if df[c].dtype=='float64']:
            df[col] = df[col].astype('int64')
        # convert object columns to str type
        for col in [c for c in df.columns if df[c].dtype=='object']:
            df[col] = df[col].astype('string')
    elif df.columns[0] == 'Continente':
        for col in [c for c in df.columns if df[c].dtype=='float64']:
            df[col] = df[col].astype('int64')
        # convert object columns to str type
        for col in [c for c in df.columns if df[c].dtype=='object']:
            df[col] = df[col].astype('string')
    print(f'4 - Converting dtypes complete\nAfter astype:\n{df.dtypes}')

    # erase non-expressed votes rows and reset index
    df.drop(df.loc[df['Candidato'] == 'VOTOS NULOS'].index, inplace=True)
    df.drop(df.loc[df['Candidato'] == 'VOTOS EN BLANCO'].index, inplace=True)
    df.reset_index(inplace=True)
    print('5 - Delete non-expressed vote complete')

    # determine the number of voting options (expressed votes only)
    if len(df)%7==0:
        vot_opt_nb = 7
    elif len(df)%2==0:
        vot_opt_nb = 2
    print('6 - Determining nb of voting options complete')

    # sort DataFrame by polling place
    sort_cols = df.columns[1:-1].to_list()
    df.sort_values(sort_cols, inplace=True)
    print('7 - Sorting complete')

    # confirm that the dataset has the expected structure
    # each candidate must appear just once in every chunk of n rows (either 7 or 2 depending on the election round)
    for i in range(vot_opt_nb):
        idx = np.arange(i,len(df),vot_opt_nb)
        assert df.iloc[idx]['Candidato'].unique() == 1
    print('8 - Checking correct structure per polling place complete')

    # create new col with total of expressed votes in each voting place
    df['Votos Exp. Mesa'] = compute_total_ballots(df['Votos TRICEL'].to_numpy(), vot_opt_nb)

    # create new col with voting percentages of each candidate in each voting place
    df['Porcentaje Mesa'] = df['Votos TRICEL'] / df['Votos Exp. Mesa'].to_numpy()
    print('9 - Adding columns with total ballots and % complete')

    # confirm that the total of votes for each candidate is correct
    for cand in df1_tot['Candidato'].unique()[:vot_opt_nb]:

        # compute the total of votes obtained by the candidate in the df
        tot_cand_votes = np.sum(df.loc[df['Candidato']==cand]['Votos TRICEL'])

        # set empty list to gather all possible scores (national or international, and first or second round)
        real_nums = []
        for df_tot in [df1_tot, df2_tot]:
            for location in ['Chile', 'Extranjero']:
                loc_total = np.sum(df_tot[df_tot['Candidato']==cand][location])
                real_nums.append(int(loc_total))
        assert int(tot_cand_votes) in real_nums


# create array of all scores for each first-round candidate
Boric = add_int_ext('GABRIEL BORIC FONT', df1_int, df1_ext)
Kast = add_int_ext('JOSE ANTONIO KAST RIST', df1_int, df1_ext)
Provoste = add_int_ext('YASNA PROVOSTE CAMPILLAY', df1_int, df1_ext)
Sichel = add_int_ext('SEBASTIAN SICHEL RAMIREZ', df1_int, df1_ext)
Artés = add_int_ext('EDUARDO ARTES BRICHETTI', df1_int, df1_ext)
Ominami = add_int_ext('MARCO ENRIQUEZ-OMINAMI GUMUCIO', df1_int, df1_ext)
Parisi = add_int_ext('FRANCO PARISI FERNANDEZ', df1_int, df1_ext)

# same for the two second-round candidates
Boric2 = add_int_ext('GABRIEL BORIC FONT', df2_int, df2_ext)
Kast2 = add_int_ext('JOSE ANTONIO KAST RIST', df2_int, df2_ext)

# create an array of the difference of expressed votes between the two rounds for each polling place
tot_exp_votes_1 = np.append(df1_int['Votos Exp. Mesa'].to_numpy(), df1_ext['Votos Exp. Mesa'].to_numpy())[::7]
tot_exp_votes_2 = np.append(df2_int['Votos Exp. Mesa'].to_numpy(), df2_ext['Votos Exp. Mesa'].to_numpy())[::2]
diff_votes_tot = tot_exp_votes_2 - tot_exp_votes_1
diff_votes_perc = np.divide(diff_votes_tot, tot_exp_votes_1, where=tot_exp_votes_1!=0)
diff_votes_perc = np.round(diff_votes_perc * 100, 2)
# narrowing range of the array to avoid outliers (check plt.hist(diff_votes_perc) to see distribution)
diff_votes_perc = np.where(diff_votes_perc < -30, -30, diff_votes_perc)
diff_votes_perc = np.where(diff_votes_perc > 55, 55, diff_votes_perc)

# confirm that the numbers checks the data of SERVEL (in total ballots table)
diff_votes_check = np.sum(df2_tot.iloc[:2]['Total'].to_numpy()) - np.sum(df1_tot.iloc[:7]['Total'].to_numpy())
assert np.sum(diff_votes_tot) == diff_votes_check

# for the national vote, create an array of the region,
# for the foreign vote, create an array of the table's length filled with same value
# join the two array and define location labels
region_array = df1_int.iloc[np.arange(0,len(df1_int), 7)]['Región'].to_numpy(dtype='<U44')
foreign_vote_array = np.full(shape=[int(len(df1_ext)/7), 1], fill_value='EXTRANJERO', dtype='<U44')
location_array = np.append(region_array, foreign_vote_array)


# set list of candidate name, in same order as the arrays so that index i will correspond to the good candidate
names = list(df1_int['Candidato'].unique())
# shorten Ominami's name
names[-2] = 'MARCO ENRIQUEZ-OMINAMI'

# create an array of seven colors for the first range of subplot (the comparison), one for each candidate
colors = cm.rainbow(np.linspace(0, 1, 7))

# CODE FOR 3-SUBPLOTS FIGURES (EN)
for i, candidate in enumerate([Boric, Kast, Provoste, Sichel, Artés, Ominami, Parisi]):
    fig, axs = plt.subplots(2, 2, figsize=[15,10])

    # extract name of the 1st round candidate
    candidate1name = names[i].title()

    # define candidate 2nd round to compare to
    if i == 1 or i == 3:
        candidate2name= 'José Antonio Kast Rist'
        candidate2 = Kast2
    else:
        candidate2name= 'Gabriel Boric Font'
        candidate2 = Boric2

    # format x and y axis in percentages
    for a, b in [(0,0), (0,1), (1,0), (1,1)]:
        axs[a][b].xaxis.set_major_formatter(PercentFormatter())
        axs[a][b].yaxis.set_major_formatter(PercentFormatter())

    # put the title in the second plot, add general description and hide the axes
    axs[0][1].annotate(text=f"2nd round behavior of\n{candidate1name}'s electorate",
                       xy=[0.5,0.8], horizontalalignment='center', fontsize=20, fontweight='bold')
    axs[0][1].annotate(text='Comparison of the results obtained at each round of the'
                            '\n2021 Chilean presidential elections (by polling station)',
                       xy=[0.5,0.6], horizontalalignment='center', fontsize=12, fontstyle='italic')
    axs[0][1].annotate('Legend:\n'
                       '1 - Expressed votes per polling station (in %)\n'
                        '2 - Electoral mobilization between the two rounds\n'
                        '3 - Vote per region',
                        xy=[0.05,0.05], horizontalalignment='left', fontsize=12, fontweight='light',
                       backgroundcolor='white', bbox=dict(edgecolor='black', facecolor='white',boxstyle='round'))
    X_max = float(max(candidate))
    Y_max = float(max(candidate2))
    axs[0][0].annotate(text='1', xy=[X_max,90], color='darkred', fontsize=20, fontweight='black')
    axs[1][0].annotate(text='2', xy=[X_max,90], color='darkred', fontsize=20, fontweight='black')
    axs[1][1].annotate(text='3', xy=[X_max,90], color='darkred', fontsize=20, fontweight='black')
    axs[0][1].axis('off')

    # set labels of the general figure
    fig.supylabel(f'{candidate2name} - 2nd round results', fontsize=16, ha='center', va='center')
    fig.supxlabel(f'{candidate1name} - 1st round results', fontsize=16, ha='center', va='center')

    # plot comparison of expressed votes in the first subplot
    sns.scatterplot(x=candidate, y=candidate2, color=colors[i], alpha=0.3, ax=axs[0][0])
    # define variables to plot national averages of candidates
    cand2_mean = float(np.nanmean(candidate2))
    cand_mean = float(np.nanmean(candidate))
    nb_pp = int(len(df1_int) / 7)
    # plot national average of elected President
    X_plot = np.linspace(0, X_max, nb_pp)
    Y_plot = np.linspace(cand2_mean, cand2_mean, nb_pp)
    axs[0][0].plot(X_plot, Y_plot, color='black', linestyle='-.', label=f'{candidate2name}\n2nd round: {round(cand2_mean,1)}%')
    # plot national average of candidate
    X_plot2 = np.linspace(cand_mean, cand_mean, nb_pp)
    Y_plot2 = np.linspace(0, Y_max, nb_pp)
    axs[0][0].plot(X_plot2, Y_plot2, color='black', linestyle=':', label=f' {candidate1name}\n1st round: {round(cand_mean, 1)}%')
    axs[0][0].legend(fontsize='small', title='National average', title_fontsize='small')

    # plot electoral mobilization in the third subplot
    sns.scatterplot(x=candidate, y=candidate2, hue=diff_votes_perc, palette='coolwarm', alpha=0.5, ax=axs[1][0])
    # Annotate total number of votes in both rounds, as well as the increase of participation in %
    axs[1][0].legend(title=f'Gain/loss of votes between\nthe two rounds (in %)', title_fontsize='small', fontsize='small')

    # plot votes according to region in the last subplot
    # a reordering the position to the north is necessary in order to create a readable heatmap
    # instantiate a list with the result of np.unique(location_array)
    regions = ['DE ANTOFAGASTA', 'DE ARICA Y PARINACOTA', 'DE ATACAMA',
       'DE AYSEN DEL GENERAL CARLOS IBAÑEZ DEL CAMPO', 'DE COQUIMBO',
       'DE LA ARAUCANIA', 'DE LOS LAGOS', 'DE LOS RIOS',
       'DE MAGALLANES Y DE LA ANTARTICA CHILENA', 'DE TARAPACA',
       'DE VALPARAISO', 'DE ÑUBLE', 'DEL BIOBIO',
       "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS", 'DEL MAULE',
       'EXTRANJERO', 'METROPOLITANA DE SANTIAGO']
    # zip the region list with a list of their respective position starting from the north
    north_to_south = [3, 1, 4, 15, 5, 12, 14, 13, 16, 2, 6, 10, 11, 8, 9, 17, 7]
    region_position = zip(regions, north_to_south)
    # create an array of the regional position of each polling place
    position_array = np.empty(len(location_array))
    for region, position in region_position:
        position_array[location_array == region] = position
    # stack all arrays of interest into a single one and sort it according to the regional position
    ordered_array = np.column_stack([candidate, candidate2, position_array])
    sorted_array = ordered_array[np.argsort(ordered_array[:,2])]
    # create plot
    sns.scatterplot(x=sorted_array[:,0], y=sorted_array[:,1], hue=sorted_array[:,2].astype('<U44'), palette='Spectral', alpha=0.4, ax=axs[1][1])
    # clean labels in order from north to south
    location_labels = ['ARICA', 'TARAPACA', 'ANTOFAGASTA', 'ATACAMA', 'COQUIMBO', 'VALPARAISO', 'METROPOLITANA',
                       "O'HIGGINS", 'MAULE', 'ÑUBLE', 'BIOBIO', 'ARAUCANIA', 'LOS RIOS', 'LOS LAGOS', 'AYSEN',
                       'MAGALLANES', 'EXTRANJERO']
    axs[1][1].legend(labels = location_labels, ncol=4, fontsize='xx-small')

    # Save and close fig
    plt.savefig(f'{candidate1name}_AllFiguresEN')
    plt.clf()