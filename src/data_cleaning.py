import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# there are many ways to normalize the data, but to run through things quickly it is easiest to just create a dataframe with normalized data
# this function will do that

def normalize_df(df):
    from sklearn import preprocessing
    norm_df = df.copy()
    col_list = df.columns.tolist().copy()
    for col in col_list:
        norm_df[f"Norm_{col}"] = preprocessing.MinMaxScaler().fit_transform(df[col].values.reshape(-1,1))
        norm_df.drop(col, axis=1, inplace= True)
    return norm_df



if __name__ == '__main__':
    
    #load Amfar opioid and HIV data, add county code
    opiod_df = pd.read_table('data/amfAR/countydata.tsv',header=0)
    opiod_df['county_code'] = opiod_df.STATEFP*1000 + opiod_df.COUNTYFP # build a county code column
    opiod_df['county_code'] = opiod_df.county_code.astype(int)

    #make changes to the amfar dataframe
    #convert from long to wide format
    index_lst = ['county_code', 'COUNTY', 'STATEABBREVIATION', 'YEAR']
    col_lst = ['INDICATOR']
    opiod_df_wide = opiod_df.pivot_table(values='VALUE', index=index_lst, columns=col_lst).reset_index()

    # Focus on just the HIV related data, from 2008 onward
    opiod_df_wide = opiod_df_wide[opiod_df_wide['YEAR'] >= 2008] # subset for years that have hiv data

    # drop unnecessary columns
    cols_to_drop = ['CDC_consult', 'vulnerable_rank']
    opiod_df_wide.drop(cols_to_drop, axis=1, inplace=True) 

    fill_nan_cols = ['HIVdiagnoses', 'HIVincidence', 'HIVprevalence','PLHIV', 'drugdeathrate', 'drugdeaths']
    opiod_df_wide[fill_nan_cols] = opiod_df_wide[fill_nan_cols].fillna(0) #fill NaNs for suppressed data with zeroes

    # Subset data to 2015
    opiod_df_15 = opiod_df_wide[opiod_df_wide['YEAR'] == 2015]
    opiod_df_15.drop(['num_SSPs', 'bup_phys', 'drugdep', 'pctunmetneed', 'nonmedpain'], axis=1, inplace=True)

    # get esimates for num_SSPs, bug_phys, drug_dep, pctunmetneed, and nonmedpain from following years

    #subset opioid related data from one year only
    #number of needle exchange programs (num_SSPs)
    opiod_df_wide_17 = opiod_df_wide[opiod_df_wide['YEAR'] == 2017]
    df_num_SSP = opiod_df_wide_17[['num_SSPs', 'county_code']]

    #number of doctors licensed to rx Buprenorphine (bup_phys)
    df_bup_phys = opiod_df_wide_17[['bup_phys', 'county_code']]

    #percent with drug dependency (drug_dep)
    opiod_df_wide_16 = opiod_df_wide[opiod_df_wide['YEAR'] == 2016]
    df_drugdep = opiod_df_wide_16[['drugdep', 'county_code']]

    #percent unmet drug treatment need (pctunmetneed)
    df_pctunmetneed = opiod_df_wide_16[['pctunmetneed', 'county_code']]

    #percent taken pain meds for nonmedical use (nonmedpain)
    df_nonmedpain = opiod_df_wide_16[['nonmedpain', 'county_code']]

    # merge these values back into 2015 dataframe
    #merge opioid related data back to the 2015 dataframe
    opiod_df_15 = opiod_df_15.merge(df_num_SSP, on='county_code')
    opiod_df_15 = opiod_df_15.merge(df_bup_phys, on='county_code')
    opiod_df_15 = opiod_df_15.merge(df_drugdep, on='county_code')
    opiod_df_15 = opiod_df_15.merge(df_pctunmetneed, on='county_code')
    opiod_df_15 = opiod_df_15.merge(df_nonmedpain, on='county_code')

    #load Men who have sex with men (MSM) estimate data
    msm_df = pd.read_csv("data/CAMP/US_MSM_Estimates_Data_2013.csv")
    msm_df['county_code'] = msm_df.STATEFP*1000 + msm_df.COUNTYFP 
    msm_df['county_code'] = msm_df.county_code.astype(int)
    msm_df['%msm12month'] = 100 * (msm_df.MSM12MTH / msm_df.ADULTMEN)
    msm_df['%msm5yr'] = 100 * (msm_df.MSM5YEAR / msm_df.ADULTMEN)   

    cols_to_drop = ['REGCODE', 'DIVCODE', 'STATEFP', 'COUNTYFP', 'CSACODE', 'CBSACODE','METDCODE', 'METMICSA', 'CENTOUTL']
    msm_df.drop(cols_to_drop, axis=1, inplace=True) #drop all unneeded columns

    #unemplyment data
    df_employment = pd.read_csv("data/ACS_14_5YR_employment/ACS_14_5YR_S2301_with_ann.csv", encoding = "ISO-8859-1", skiprows=1)
    df_employment = df_employment[['Id2', 'Unemployment rate; Estimate; Population 16 years and over']]
    df_employment.columns = ['county_code', 'unemployment_rate']

    #poverty data
    df_poverty = pd.read_csv("data/ACS_14_5YR_poverty/ACS_14_5YR_S1701_with_ann.csv", encoding = "ISO-8859-1", skiprows=1)
    df_poverty = df_poverty[['Id2', 'Percent below poverty level; Estimate; Population for whom poverty status is determined']]
    df_poverty.columns = ['county_code', 'poverty_rate']
    df_poverty.head()


    #income data
    df_income_full = pd.read_csv("data/ACS_14_5YR_income/ACS_14_5YR_S1901_with_ann.csv", encoding = "ISO-8859-1", skiprows=1)
    df_income = df_income_full[['Id2', 'Households; Estimate; Total']]
    df_income.columns = ['county_code', 'household_income']

    #merge asfAR hiv/opioid data with CAMP MSM data
    df_main = opiod_df_15.merge(msm_df, on='county_code')

    #merge in ACS data
    df_main = df_main.merge(df_employment, on='county_code')
    df_main = df_main.merge(df_poverty, on='county_code')
    df_main = df_main.merge(df_income, on='county_code')

    main_cols = df_main.columns

    # we want to add the Household Mean and Median income to the df_main

    income_cols = df_income_full.columns.tolist()
    df_main = df_main.merge(df_income_full[['Households; Estimate; Mean income (dollars)','Households; Estimate; Median income (dollars)']], how= 'outer', left_on= df_main.index, right_on= df_income_full.index)
    df_main.drop('key_0', axis=1, inplace= True)
    new_cols = ['county_code', 'COUNTY', 'STATEABBREVIATION', 'YEAR', 'AMAT_fac',
           'HIVdiagnoses', 'HIVincidence', 'HIVprevalence', 'MH_fac',
           'Med_AMAT_fac', 'Med_MH_fac', 'Med_SA_fac', 'Med_SMAT_fac',
           'Med_TMAT_fac', 'PLHIV', 'Population', 'SA_fac', 'SMAT_fac', 'TMAT_fac',
           'drugdeathrate', 'drugdeathrate_est', 'drugdeaths', 'mme_percap',
           'partD30dayrxrate', 'pctunins', 'num_SSPs', 'bup_phys', 'drugdep',
           'pctunmetneed', 'nonmedpain', 'ADULTMEN', 'MSM12MTH', 'MSM5YEAR',
           '%msm12month', '%msm5yr', 'unemployment_rate', 'poverty_rate',
           'household_income','Mean income','Median income']
    df_main.columns = new_cols

    # lets save this as a single file
    df_main = pd.read_excel('data/combined_df.xlsx', index_col=0)

    # removing columns that we don't want to use for modeling
    drop_list = ['PLHIV', 'HIVdiagnoses', 'drugdeathrate_est', 'drugdeaths']
    df_main.drop(drop_list, axis=1, inplace= True)

    # lets drop those rows that are all NaN
    nan_rows = df_main[df_main['county_code'].isna() == True].index
    df_main.drop(nan_rows, axis=0, inplace= True)

    # lets fill the rest of those NaN with zeros
    df_main.fillna(0, inplace= True)

    # to do that, need to drop rows with no HIVprevalence
    df_main = df_main[df_main['HIVprevalence']> 0]
    df_main['HIV_rel_infection'] = (df_main['HIVincidence']/ df_main['HIVprevalence']) * 100

    # what does the distribution of this look like?
    sns.distplot(df_main['HIV_rel_infection']);

    # how many values are there?
    print(df_main['HIV_rel_infection'].count())

    # how many VERY high values are there?
    print(df_main['HIV_rel_infection'][df_main['HIV_rel_infection']>15].count())

    # dropping those high df_main['HIV_rel_infection'] values
    df_main = df_main[df_main['HIV_rel_infection'] <= 15]

    # because the drug treatment facilities seem to have similar information, we want to combine them for a single count
    # keeping those that accept medicaid separate
    df_main['Num_drug_treatment_fac'] = (df_main['AMAT_fac'] + df_main['SA_fac'] + df_main['SMAT_fac'] + df_main['TMAT_fac'])
    df_main['Num_med_drug_treatment_fac'] = (df_main['Med_AMAT_fac'] + df_main['Med_SA_fac'] + df_main['Med_SMAT_fac'] + df_main['Med_TMAT_fac'])

    # now we can drop those original columns
    drop_fac= ['AMAT_fac','Med_AMAT_fac','Med_SA_fac','Med_SMAT_fac','Med_TMAT_fac','SA_fac','SMAT_fac','TMAT_fac']
    df_main.drop(drop_fac, axis=1, inplace= True)

    # how many of the facilities accept medicaid?
    accept_med = len((df_main[df_main['Num_drug_treatment_fac'] == df_main['Num_med_drug_treatment_fac']]))
    print(accept_med)

    # how many don't?
    dont_accept_med = len((df_main[df_main['Num_drug_treatment_fac'] != df_main['Num_med_drug_treatment_fac']]))
    print(dont_accept_med)

    print("Percent that accept medicaid: {:.2f}%".format(100* accept_med/(accept_med + dont_accept_med)))

    # I'm curious if the percent of men in the population varies
    df_main['Percent_men'] = (df_main['ADULTMEN']/df_main['Population']) * 100

    # Does the ratio of income metrics have an impact?
    # a value close to zero would indicate a uniform income distribution
    # lower values indicate more income inequality
    df_main['income_ratio'] = df_main['Median income']/df_main['Mean income']

    # Does the relative amount of MSM activity within the year vs 5year window have any impact?
    df_main['MSM_recent_ratio'] = (df_main['%msm12month']/df_main['%msm5yr']) * 100

    # lets select which columns we want to use as predictors
    all_cols = df_main.columns.tolist()
    X_cols = [x for x in all_cols if x not in ['COUNTY','STATEABBREVIATION','YEAR','HIVincidence', 'HIVprevalence','HIV_rel_infection']]

    # lets create our X and y as numpy arrays
    X = df_main[X_cols]
    y = df_main['HIV_rel_infection']

    # now the DF looks pretty clean. Lets do a train/test split and keep them as pandas DF
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    df_pos_inf = df_main[df_main['HIV_rel_infection'] > 0]
    print("after dropping zeros, there are {} rows".format(df_pos_inf.shape[0]))
    print("and before there were {} rows".format(df_main.shape[0]))
    print("Thats a {:.2f}% reduction!".format(100 * (df_main.shape[0]-df_pos_inf.shape[0])/df_main.shape[0]))

    X2 = df_pos_inf[reduced_feature_list]
    y2 = df_pos_inf['HIV_rel_infection']
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2)

    # apply normalize on data
    # not that we are applying the log-transformationt to the data BEFORE normalization in order to avoid 0 values
    y2 = pd.DataFrame(y2)
    y2 = y2[y2['HIV_rel_infection']>0]
    X2_norm = normalize_df(X2)
    y2_norm = normalize_df(np.log(y2))

    # split the normalized X2 df
    X2_norm_train, X2_norm_test, y2_norm_train, y2_norm_test = train_test_split(X2_norm, y2_norm)