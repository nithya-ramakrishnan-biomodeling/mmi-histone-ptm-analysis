import pandas as pd
import operator
import numpy as np
from typing import  Literal, Union, Tuple
from itertools import combinations
import json


operators = {
        "lt": operator.lt,  # Less than 
        "gt": operator.gt,  # greater than
        "le" : operator.le, # less than or equal to
        "ge" : operator.ge, # greater than or equal to
        "eq" : operator.eq, # equal to 
        "ne" : operator.ne, } # not equal to


def cond_comparisonr(value: float,
                    opt:Union[Literal['lt', 'gt', 'le', 'ge', 'eq', 'ne'] ] ,
                                threshold:float ) -> bool:
    """ Comapres input value based on given threshold and  operator
    * Note:  Read the execution of this function in the parameter order.
    * Eg: 1 lt (less than) 3 -> True


    Parameters
    ----------
    value : float
        input value to check.
    opt : Name of the operator
        
    threshold : float
        Threshold value.

    Returns
    -------
    _type_
        _description_
    """
    
    return operators[opt](value, threshold)


def masked_df_gen(df:pd.DataFrame, threshold_value: tuple, oprtr: 
                  Union[Literal['lt', 'gt', 'le', 'ge', 'eq', 'ne'] ]) -> pd.DataFrame:
    """ Masking dataframe by giving the threshold value. 
    * Eg. Getting the dataframe values greater than 5:
            df> 5

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    cond : tuple
        Threshold values
        
    oprtr : Union[Literal['lt', 'gt', 'le', 'ge', 'eq', 'ne'] ]

    Returns
    -------
    pd.Dataframe
        Masked datframe after removing thr threshold condtion.
    """


    if len(threshold_value)==1 and len(oprtr)==1:
        print(threshold_value, oprtr)
        return df[cond_comparisonr(df, oprtr[0], threshold_value[0] )]

    elif len(threshold_value)==2 and len(oprtr)==2:

        left_threshold, right_threshold = threshold_value
        left_optr, right_optr = oprtr
        left_mask  = cond_comparisonr(df, left_optr, left_threshold)
        right_mask  = cond_comparisonr(df,right_optr, right_threshold)
        masked_comb_df = df[left_mask & right_mask]
        return  masked_comb_df
    
    else:
        print(f"Please provide the correct parameter values.")
        print(f"This is what I got from the input operator: {oprtr} and Threshold:{threshold_value} values.")


def two_mi_value_combn_three_mi_value_gen(three_mi_df: pd.DataFrame, 
                                          two_mi_df:pd.DataFrame
                                          , cond_value: Union[
                                              Tuple [Literal['lt', 'gt', 'le', 'ge', 'eq', 'ne'] ],
                                              Literal['lt', 'gt', 'le', 'ge', 'eq', 'ne']], 
                                              oprtr: Tuple[tuple, tuple],
                                              cov_df:pd.DataFrame=None,
                                              corr_df:pd.DataFrame=None,) -> dict:
    


    unique_three_var_combn = list(combinations(two_mi_df.keys(), r=3))
    three_mi_neg_two_mi_pos_dct = {}

    for (one, two, three) in unique_three_var_combn:
        two_mi_three_var_value = {}
        
        # getting the all the possible two variable combinations:
        two_var_combn = np.array([two_mi_df[col][indx] for col, indx in  combinations([one, two, three], r=2)])
        # no_elemnts = len(two_var_combn[two_var_combn>0])
        three_mi = three_mi_df[one][f"{two}_{three}"]

        if len(cond_value)==2:
            lt, gt = oprtr
            lt_oprtor, right_oprtr = cond_value
    
            if cond_comparisonr(float(three_mi), lt_oprtor, lt)  and cond_comparisonr(float(three_mi), right_oprtr, gt):
                two_mi_three_var_value[f"{str(3)}_mi"] = three_mi
                two_mi_three_var_value[f"{str(2)}_mi"] = {(col, indx):two_mi_df[col][indx] for col, indx in  combinations([one, two, three], r=2)}

                if isinstance(cov_df, pd.DataFrame): 
                    two_mi_three_var_value["cov"] = {(col, indx):cov_df[col][indx] for col, indx in  combinations([one, two, three], r=2)}
                
                if isinstance(corr_df, pd.DataFrame): 
                    two_mi_three_var_value["corr"] = {(col, indx):corr_df[col][indx] for col, indx in  combinations([one, two, three], r=2)}

                three_mi_neg_two_mi_pos_dct[(one, two, three)] = two_mi_three_var_value

        elif len(cond_value)==1:
            no_elemnts = len(two_var_combn[two_var_combn>0])

            if no_elemnts >0 and cond_comparisonr(float(three_mi), cond_value[0], oprtr[0]):
                two_mi_three_var_value[f"{str(3)}_mi"] = three_mi
                two_mi_three_var_value[f"{str(2)}_mi"] = {(col, indx):two_mi_df[col][indx] for col, indx in  combinations([one, two, three], r=2)}

                if isinstance(cov_df, pd.DataFrame): 
                    two_mi_three_var_value["cov"] = {(col, indx):cov_df[col][indx] for col, indx in  combinations([one, two, three], r=2)}
                
                if isinstance(corr_df, pd.DataFrame): 
                    two_mi_three_var_value["corr"] = {(col, indx):corr_df[col][indx] for col, indx in  combinations([one, two, three], r=2)}

                three_mi_neg_two_mi_pos_dct[(one, two, three)] = two_mi_three_var_value

    return  three_mi_neg_two_mi_pos_dct


def df_and_text_generator(input_dict:dict,
                           df_new_indx:list, 
                          df_new_col: list):   
    
     
    # new pandas dataframe for: filtered three mi values, & hover text generation.
    three_mi_neg_two_mi_pos_df = pd.DataFrame(index=df_new_indx, columns= df_new_col)
    three_mi_neg_two_mi_pos_df_hover = pd.DataFrame(index=df_new_indx, columns= df_new_col)

    for  i in input_dict.items():

        col = i[0][0]   
        indx = f"{i[0][1]}_{i[0][2]}"
        
        orginal_dict = i[1][f"{str(2)}_mi"]
        three_mi_neg_two_mi_pos_df.loc[indx, col ] = i[1][f"{str(3)}_mi"]

        cov_dict = { "Cov" :  {str(key): round(value,3) for key, value in i[1]["cov"].items()} }
        corr_dict = {"Corr": {str(key): round(value,3) for key, value in i[1]["corr"].items()}}
        # Doing json dump for transforming the dictionary into string format.
        converted_dict = {f"{str(2)}_mi": {str(key): round(value,3) for key, value in orginal_dict.items()}}

        combind_dict = converted_dict| cov_dict | corr_dict # Combining the dictioanry.

        # List of characters to remove
        unwanted_chars = ['"', r"{",r"}", " ",r")", r"(" ]

        # Function to clean and format JSON for hover text
        def clean_json(json_string, unwanted_chars):
            for char in unwanted_chars:
                json_string = json_string.replace(char, '')
            return json_string.replace('\n', '<br>')
        
        three_mi_neg_two_mi_pos_df_hover.loc[indx, col ] = [clean_json(json.dumps( combind_dict, indent=2), unwanted_chars=unwanted_chars),]

    three_mi_neg_two_mi_pos_df_row = three_mi_neg_two_mi_pos_df.dropna(axis=0,  how="all")  # Dropping the rows, if all are empty.
    three_mi_neg_two_mi_pos_df_col = three_mi_neg_two_mi_pos_df_row.dropna(axis=1, how="all") # Dropping colums, if all are empty.
    three_mi_neg_two_mi_pos_df_row_hover = three_mi_neg_two_mi_pos_df_hover.dropna(axis=0,  how="all")  # Dropping the rows, if all are empty.
    three_mi_neg_two_mi_pos_df_col_hover = three_mi_neg_two_mi_pos_df_row_hover.dropna(axis=1, how="all") # Dropping the rows, if all are empty.
    additional_info = three_mi_neg_two_mi_pos_df_col_hover.to_numpy()
    return three_mi_neg_two_mi_pos_df_col, additional_info
