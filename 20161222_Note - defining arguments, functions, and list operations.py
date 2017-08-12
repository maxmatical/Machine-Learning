# defining a function
########################################
# def function(argument, argument):
    # blah blah blah
    # return
    # pass
# note: don't have to enter type of argument


# operations on elements in a list
#######################################
list_1 = [1,2,3]
# method 1: using for loops
# for loop element i in list
def add_1 (list):
    new_list = [] # create empty list
    for i in list:
        new_list.append(i+1) # populate the new empty list with new values
    return new_list # return the new value
add_1(list_1)    
    
# method 2: no need to append
def subtract_1(list):
    new_list = [i-1 for i in list] # operation to every element in the list
    return new_list
subtract_1(list_1) 

# note: always define a new list/value and return that new list/value


# working with arrays
#######################################

# 1) taking log (ln) of column k of an array in a dataframe df
k = 2
np_df[:,2] = [np.log(i) for i in np_df[:,k]]
