### imports
import re
import statistics
import multiprocessing
import random
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import numpy
from collections import Counter
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import os
import math
from statistics import median


#####*************************

########## FUNCTIONS ######## DO NOT MODIFY

####*************************



def retrieve_indiv_from_logbook(input_list, x):
    unique_elements = []
    for sublist in reversed(input_list):
        for item in sublist:
            if item not in unique_elements:
                unique_elements.append(item)
                if len(unique_elements) == x:
                    return unique_elements
    return unique_elements[:x]


def find_median_indexes(data):
    # Transpose the data to get elements at each index
    transposed_data = [[lst[i] for lst in data] for i in range(len(data[0]))]

    # Compute the median for each element at each index
    medians = [median(lst) for lst in transposed_data]

    # Find the index of the median values
    median_indexes = [lst.index(median_val) for lst, median_val in zip(transposed_data, medians)]

    return medians, median_indexes


def find_average_indexes(data):
    # Transpose the data to get elements at each index
    transposed_data = [[lst[i] for lst in data] for i in range(len(data[0]))]

    # Compute the average for each element at each index
    averages = [sum(lst) / len(lst) for lst in transposed_data]

    return averages


def compute_High(lst):
    # Transpose the list of lists
    transposed = list(zip(*lst))

    # Find the highest value for each column
    highest_values = []
    highest_indices = []
    for column in transposed:
        max_value = max(column)
        highest_values.append(max_value)
        highest_index = column.index(max_value)
        highest_indices.append(highest_index)

    return highest_indices, highest_values


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_individual(individuals, best_point):
    closest_individual = None
    min_distance = float('inf')
    individuals_with_distance = []

    for ind in individuals:
        point = (ind.fitness.values[0], ind.fitness.values[1])
        # point = ( 0, ind.fitness.values[1])  # Use only obj1 value
        distance = euclidean_distance(point, best_point)
        individuals_with_distance.append((ind, distance))

    individuals_with_distance.sort(key=lambda x: x[1])  # Sort by distance
    # print(individuals_with_distance)
    closest_individuals = [x[0] for x in individuals_with_distance]
    return closest_individuals


def append_string_IaC(IaCFile, suffix):
    result = IaCFile + suffix
    return result


def get_nb_files_matrix(path_sim_matrix):
    with open(path_sim_matrix, newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix = [row for row in reader]
        num_cols = len(matrix[0])
    return num_cols


def get_query_file_index(file, path_sim_matrix):
    with open(path_sim_matrix, newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix = [row for row in reader]

    suffix = '_(IaC)_'
    base_path = Git_path
    relative_path = file.replace(base_path, '')
    IaC_file = append_string_IaC(relative_path, suffix)
    index_IaC_file = matrix[0].index(IaC_file)
    # print(index_IaC_file)
    return index_IaC_file


def compute_MRR(recommended_ranked_lists, truths):
    reciprocal_ranks = []

    for truth, recommended_list in zip(truths, recommended_ranked_lists):
        rank = next((i + 1 for i, item in enumerate(recommended_list) if item in truth),
                    0)
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)

    if reciprocal_ranks:
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    else:
        mrr = 0

    return mrr


def get_files_(path_sim_matrix):
    with open(path_sim_matrix, newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix = [row for row in reader]

    # Put the first value of each row in a list
    first_values = [row[0] for row in matrix]
    # print("First values:", first_values)
    return first_values





def retrieve_data_from_csv(csv_file, index_list):
    df = pd.read_csv(csv_file, header=None)
    # print(df)

    data = df.iloc[index_list, 0].tolist()
    # print(data)
    return data


def nb_solutions(individual):
    value = len(individual)
    # print(value)
    fitness = 1 / (1 + value)  # in case ind has no eleent, avoid division by 0
    fitness = value
    return fitness

def create_individual():
    # Generate a random size for the individual between 3 and 10
    size = random.randint(min_recommendations, max_recommendations)

    # Create a set of indices
    indices = set()
    while len(indices) < size:
        r = random.randint(1, num_files - 1)
        if r != index_IaC_file:
            indices.add(r)
    return creator.Individual(list(indices))


def evaluate(individual):
    similarities1 = 0.0
    similarities2 = 0.0
    similarities3 = 0.0

    index = 0
    for file2 in individual:
        # print(file2)
        # print(index_IaC_file - 1)
        similarity1 = content_sim_data[index_IaC_file- 1, file2 ]

        similarities1 += similarity1 * ((len(individual) - index) / len(individual))
        #print(similarity1 * ((len(individual) - index) / len(individual)))

        similarity2 = path_sim_data[index_IaC_file- 1, file2 ]
        similarities2 += similarity2 * ((len(individual) - index) / len(individual))

        similarity3 = co_change_data[index_IaC_file- 1, file2 ]
        similarities3 += similarity3 * ((len(individual) - index) / len(individual))

        index += 1

    #print(similarities1)
    avg_similarities1 = np.sum(similarities1)
    #print(avg_similarities1)
    avg_similarities2 = np.sum(similarities2)

    avg_similarities = (avg_similarities1 + avg_similarities2) / 2

    # avg_similarities = avg_similarities / len(individual)
    avg_sim = avg_similarities/ len(individual)
    avg_similarities3 = similarities3/ len(individual)

    avg_similar = (avg_sim + avg_similarities3) / 2

    FSIM = avg_similar


    if len(set(individual)) != len(individual):
        return 0.0,
    else:
        return FSIM,


class IndividualHashable:
    def __init__(self, individual):
        self.individual = individual

    def __hash__(self):
        return hash(tuple(self.individual))

    def __eq__(self, other):
        return isinstance(other, IndividualHashable) and self.individual == other.individual


import random


def modify_solution(current_solution,step_size,temperature):
    if len(current_solution) <= 1:
        # If there's only one file or none, no modification can be made
        return current_solution

    new_solution = current_solution.copy()  # Make a copy to avoid altering the original solution
    # Select two indices at random to swap
    print(temperature)
    if temperature < 1e-100:
        modification_type = 'shift'  # Prefer shifting as the algorithm converges
    else:
        # Skew the choice towards 'swap' in the earlier stages
        modification_type = 'swap'

    if modification_type == 'swap':
        idx1, idx2 = random.sample(range(len(new_solution)), 2)
        # Swap the files
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]

    elif modification_type == 'shift':
        index_to_modify = random.randint(0, len(new_solution)-1)

        # Modify the selected index within the bounds
        new_solution[index_to_modify] = min(max(1, new_solution[index_to_modify] + random.randint(-step_size, step_size)),
                                                num_files - 1)

    return new_solution


def main(evaluate, create_individual, initial_temp, cooling_rate, num_iterations,f,initial_step_size):
        # Initialize with a random solution
        current_solution = create_individual()
        current_fitness = evaluate(current_solution)[0]  # Assuming evaluate returns a tuple
        best_solution = current_solution
        best_fitness = current_fitness
        fitness_values = [best_fitness]
        print("start point= ", best_fitness)

        temperature = initial_temp
        step_size = initial_step_size  # Initialize step size
    #with open(Project_path + project + '/' + Algo +'/'+'fitness_over_iterations.txt', 'a') as file:
        #file.write(f"{f}\n")
        for iteration in range(num_iterations):
            candidate_solution = modify_solution(current_solution,step_size,temperature)
            candidate_fitness = evaluate(candidate_solution)[0]
            E = abs(candidate_fitness-best_fitness)

            # Calculate acceptance probability
            if candidate_fitness > current_fitness:
                # Directly accept better solutions
                current_solution = candidate_solution
                current_fitness = candidate_fitness

                # Update the best solution found so far, if applicable
                if candidate_fitness > best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
                    fitness_values.append(best_fitness)
            else:
                # Calculate acceptance probability for worse solutions
                acceptance_probability = math.exp((candidate_fitness - current_fitness) / temperature)
                if acceptance_probability > random.random():
                    current_solution = candidate_solution
                    current_fitness = candidate_fitness

            # Cool down the temperature
            temperature *= cooling_rate
            #step_size *= cooling_rate

            #print(f"Iteration {iteration}: Best Fitness {best_fitness}")
            #file.write(f"{iteration}, {best_fitness}\n")
    # Plotting the fitness values
   # plt.plot(fitness_values, 'ro-')
   # plt.xlabel('Iteration')
   # plt.ylabel('Best Fitness Value')
   # plt.show()
        return best_solution, best_fitness


if __name__ == "__main__":
    projects = ['puppet-openstack-integration']
    Algo = 'SA'
    initial_temp = 10000
    cooling_rate = 0.99
    initial_step_size=50
    for project in projects:

        # 'matrix-docker-ansible-deploy', 'eyp-systemd', 'control-repo'
        random.seed(64)

        # Process Pool
        cpu_count = multiprocessing.cpu_count()
        # print(f"CPU count: {cpu_count}")
        pool = multiprocessing.Pool(8)
        toolbox = base.Toolbox()

        toolbox.register("map", pool.map)
        PRECISION = []
        RECALL = []
        MRR = []

        FSCORE = []
        pareto_front_indiv = []
        pareto_front_fitness = []
        pareto_front_fitness_values = []

       

        Git_path = ''
        Project_path = ''
        
        matrix = "Matrices_90_10"
        content_sim_matrix = Project_path + project + '/' + matrix + '/content_sim_matrix.csv'
        path_sim_matrix = Project_path + project + '/' + matrix + '/normalized_path_matrix.csv'
        co_change_matrix = Project_path + project + '/' + matrix + '/normalized_co_change_matrix.csv'
        IaC_files = Project_path + project + '/' + 'IaC_Files_Antecedent/'


        files = get_files_(path_sim_matrix)



        # # Load the matrix from the CSV file using numpy
        content_sim_data = np.genfromtxt(content_sim_matrix, delimiter=',', skip_header=1)
        path_sim_data = np.genfromtxt(path_sim_matrix, delimiter=',', skip_header=1)
        co_change_data = np.genfromtxt(co_change_matrix, delimiter=',', skip_header=1)
        # # Replace "nan" values with 0.0
        content_sim_data = np.nan_to_num(content_sim_data, nan=0.0)
        path_sim_data = np.nan_to_num(path_sim_data, nan=0.0)
        co_change_data = np.nan_to_num(co_change_data, nan=0.0)


        Fig_ = "Generations_"
        Stats_ = "Stats_"
        _31_runs = "31_runs"
        Results_Median = "Results_Median"
        Precision = 'Precision_based'
        Recall = 'Recall_based'
        Fscore = 'Fscore_based'
        SR = 'Success_Rate_based'
        MRR = 'MRR_based'
        Results_Commit_level = 'Results_Commit_level'

        tops = [1,3, 5, 7, 10]
        
        median_indexes = ['Precision_based', 'Recall_based', 'MRR_based',
                          'Fscore_based', 'Success_Rate_based']

        os.mkdir(
            Project_path + project + '/' + Algo + '/')

        for top in tops:
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/')

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Precision)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Recall)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Fscore)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + SR)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + MRR)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Precision + '/' + Results_Median)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Recall + '/' + Results_Median)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Fscore + '/' + Results_Median)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + SR + '/' + Results_Median)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + MRR + '/' + Results_Median)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Precision + '/' + Results_Commit_level)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Recall + '/' + Results_Commit_level)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Fscore + '/' + Results_Commit_level)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + SR + '/' + Results_Commit_level)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + MRR + '/' + Results_Commit_level)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Precision + '/' + _31_runs)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Recall + '/' + _31_runs)

            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + Fscore + '/' + _31_runs)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + SR + '/' + _31_runs)
            os.mkdir(
                Project_path + project + '/' + Algo + '/' + str(
                    top) + '/' + MRR + '/' + _31_runs)

        os.mkdir(Project_path + project + '/' + Algo + '/' + Fig_)
        os.mkdir(Project_path + project + '/' + Algo + '/' + Stats_)
        

        SUCESSRATELIST = []

        #pop_size = 500

        max_recommendations = 20
        min_recommendations = 10
        num_generations = 300*500

        L1 = []
        L2 = []
        L3 = []
        L4 = []
        L5 = []
        L6 = []
        L7 = []
        L8 = []
        L9 = []
        L10 = []
        indexes = []
        List_iaC_Files = []
        SUCESSRATE_File = []
        for file in files:
            if '_(IaC)_' in file:
                # print(files[1])
                # print(file)
                List_iaC_Files.append(file)
        # counter=-1
        # print(List_iaC_Files)
        for counter in range(0, len(List_iaC_Files)):
            print('counter = ', counter)
            file = List_iaC_Files[counter]
            filenameZ = os.path.basename(file)
            last_folder = os.path.basename(os.path.dirname(file))
            filenameX = last_folder + "-" + filenameZ+str(counter)

            if '_(IaC)_' in file:
                file = file.replace('_(IaC)_', '')
                print(file)
                # print(List_iaC_Files[counter])

                query_IaC_file_path = Git_path + file
                index_IaC_file = get_query_file_index(query_IaC_file_path, path_sim_matrix)

                file_path = Project_path + project + '/_10_test_Of_Commits.csv'
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None)
                    Co_changed_Files_in_commit = []
                    Index_hash = []

                    # loop through the rows in the DataFrame
                    for index, row in df.iterrows():
                        List_Files = []
                        Index_hash.append(index)
                        # loop through the columns in the row
                        for column_index, column_value in enumerate(row):
                            List_Files.append(column_value)

                        # print(List_Files)
                        new_list = [x for x in List_Files if str(x) != 'nan']
                        Co_changed_Files_in_commit.append(new_list)

                # Example usage:
                file_path_hash = Project_path + project + '/_10_test_Of_Commits_hash.csv'

                indexes_to_retrieve = Index_hash
                all_commits_hash = retrieve_data_from_csv(file_path_hash, indexes_to_retrieve)

                # print(Co_changed_Files)
                base_path = Git_path
                query_IaC_file_path = query_IaC_file_path.replace(base_path, "")
                query_IaC_file_path = query_IaC_file_path + '_(IaC)_'
                # print(query_IaC_file_path)

                ## retrive only commits with the query file

                commits_with_iac_file_20 = [lst for lst in Co_changed_Files_in_commit if query_IaC_file_path in lst]
                commits_with_iac_file_indexes = [index for index, lst in enumerate(Co_changed_Files_in_commit) if
                                                 query_IaC_file_path in lst]

                selected_commits_hash = [all_commits_hash[index] for index in commits_with_iac_file_indexes]

                # print(selected_commits_hash)

                matching_indexes = [index for index, sublist in enumerate(commits_with_iac_file_20) if
                                    all(item == query_IaC_file_path for item in sublist)]
                for index in sorted(matching_indexes, reverse=True):
                    selected_commits_hash.pop(index)

                for sublist in commits_with_iac_file_20:
                    if query_IaC_file_path in sublist:
                        sublist.remove(query_IaC_file_path)

                count_20 = len([lst for lst in commits_with_iac_file_20 if len(lst) > 0])
                # print(count_20)

                file_path90 = Project_path + project + '/_90_train_Of_Commits.csv'
                if os.path.exists(file_path90):
                    df = pd.read_csv(file_path90, header=None)
                    Co_changed_Files_in_commit90 = []

                    # loop through the rows in the DataFrame
                    for index, row in df.iterrows():
                        List_Files90 = []
                        # loop through the columns in the row
                        for column_index, column_value in enumerate(row):
                            # do something with the column value
                            # print(f'Row {index}, Column {column_index}: {column_value}')
                            List_Files90.append(column_value)

                        # print(List_Files)
                        new_list = [x for x in List_Files90 if str(x) != 'nan']
                        Co_changed_Files_in_commit90.append(new_list)

                commits_with_iac_file_90 = [lst for lst in Co_changed_Files_in_commit90 if
                                            query_IaC_file_path in lst]
                # print(commits_with_iac_file_20)

                for sublist in commits_with_iac_file_90:
                    if query_IaC_file_path in sublist:
                        sublist.remove(query_IaC_file_path)

                count_90 = len([lst for lst in commits_with_iac_file_90 if len(lst) > 0])
                # print(count_90)

                #
                if count_20 > 0 and count_90 > 0:

                    _31_Tops_precision = []
                    _31_Tops_recall = []
                    _31_Tops_fscore = []
                    _31_Tops_MRR = []
                    _31_Tops_SR = []
                    _31_Tops_BestSolution = []
                    _31_Tops_comodified_files = []
                    _31_Tops_nb_comodified_files = []
                    _31_Tops_recommended_files = []
                    _31_Tops_iac_file = []
                    _31_Tops_nb_recommended_files = []
                    _31_Tops_commit_hash = []

                    _31_median_solution = []
                    _31_median_recommended_files = []
                    _31_highest_solution = []
                    _31_highest_recommended_files = []

                    os.mkdir(Project_path + project + '/' + Algo + '/' + Stats_ + '/' + filenameX)
                    os.mkdir(Project_path + project + '/' + Algo + '/' + Fig_ + '/' + filenameX)

                    for iter in range(31):

                        print('##################################')

                        print('### ***** RUN ', str(iter), '***** ###')

                        print('##################################')

                        Tops_precision = []
                        Tops_recall = []
                        Tops_fscore = []
                        Tops_MRR = []
                        Tops_SR = []
                        Tops_BestSolution = []
                        Tops_comodified_files = []
                        Tops_nb_comodified_files = []
                        Tops_recommended_files = []
                        Tops_iac_file = []
                        Tops_nb_recommended_files = []
                        Tops_commit_hash = []

                        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                        creator.create("Individual", list, fitness=creator.FitnessMax)

                        num_files = get_nb_files_matrix(path_sim_matrix)

                        toolbox.register("individual", create_individual)

                        # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.file_index, n=num_recommendations)

                        # Define the population generator: creates a population of random individuals

                        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                        toolbox.register("evaluate", evaluate)

                        ind1 = toolbox.individual()

                        ind1.fitness.values = evaluate(ind1)


                        ###*********GENERATION*********#######

                        # scoop.futures.scoop(main)

                        best_ind, best_fitness = main(evaluate, create_individual, initial_temp, cooling_rate,num_generations,file,initial_step_size)
                        ind=[]
                        print(best_ind,best_fitness)
                        ind.append(best_ind)


                        filenamegen = f'{iter}_iteration.csv'

                        data = {'IaC file': query_IaC_file_path, 'Iteration':iter,'best_fitness': best_fitness,
                                'Best_individual': ind}
                        df = pd.DataFrame(data)

                        df.to_csv(
                            Project_path + project + '/' + Algo + '/' + Stats_ + '/' + filenameX + '/' + filenamegen,
                            index=False)
                        # # create a figure with two subplots
                        #
                        # extract statistics:

                        topk = 0
                        for top in tops:
                            topk = top

                            _31_Top_best_solution_commit = []
                            _31_precision_commit = []
                            _31_recall_commit = []
                            _31_Fscore_commit = []
                            _31_MRR_commit = []
                            _31_SuccessScore_commit = []
                            _31_IaC_file_commit = []

                            _31_Co_modified_files_commit = []
                            _31_Nb_co_modified_files_commit = []
                            _31_Recommended_files_commit = []
                            _31_Nb_recommended_files_commit = []

                            _31_best_solution = []
                            _solution = []
                            _31_Top_best_solution = []
                            _31_precision = []
                            _31_recall = []
                            _31_Fscore = []
                            _31_MRR = []
                            _31_SuccessScore = []
                            _31_IaC_file = []
                            _31_Co_modified_files = []
                            _31_Nb_co_modified_files = []
                            _31_Recommended_files = []
                            _31_Nb_recommended_files = []
                            _31_commits_hash = []



                            print("top", topk)

                            #best_ind1 = tools.selBest(best_individuals, 1)[-1]
                            #print("best_individual is: ", best_ind1)
                            #print("best_individual fitness functions scores: ", best_ind1.fitness.values)

                            #print(best_individuals)



                            print("best_individual is:", best_ind)
                            # print("best_individual fitness functions scores:", best_ind.fitness.values)

                            recommendations = [files[i] for i in best_ind]
                            # print(recommendations)
                            # print("[{}] best_score: {}".format(logbook[-1]['gen'], logbook[-1]['min'][0]))
                            # fitness_values = best_ind.fitness.values

                            ### EVALUATION ON TEST SET #####

                            path_lists = []
                            recommended_file_paths_80 = []

                            path_lists = [files[i] for i in best_ind]
                            recommended_file_paths_80.append(path_lists)

                            ground_truth = commits_with_iac_file_20
                            recommended_items = [item for sublist in recommended_file_paths_80 for item in sublist]

                            # Count the occurrences of each element in the list
                            counted_list = Counter(recommended_items)

                            ranked_list = [element for element, count in counted_list.most_common()]

                            recommended_ranked_list_top = ranked_list[:top]

                            precision_list = []
                            recall_list = []
                            mrr_list = []
                            fscore_list = []
                            ground_truth_non_empty = [lst for lst in ground_truth if lst]
                            # print('ground_truth', ground_truth_non_empty)

                            Common_Elements = []
                            PRECISION_File = []
                            SUCESSRATE_File = []
                            RECALL_File = []
                            MRR_File = []
                            FSCORE_File = []
                            recommended = []
                            Truthset = []

                            for truth in ground_truth_non_empty:

                                # print("truth",len(truth))
                                # print('tops:', len(recommended_ranked_list_top))
                                recommended.append(len(recommended_ranked_list_top))

                                Truthset.append(len(truth))

                                _31_Nb_co_modified_files_commit.append(len(truth))
                                _31_Nb_recommended_files_commit.append(len(recommended_ranked_list_top))
                                _31_IaC_file_commit.append(query_IaC_file_path)
                                _31_Recommended_files_commit.append(recommended_items)

                                _31_Co_modified_files_commit.append(truth)

                                # Get the recommended items from the individual
                                # recommended_items = set([i for i, r in enumerate(ind) if r > threshold])
                                # recommended_items = ind
                                # print(recommended_items)
                                # Compute precision and recall

                                common_elements = set(truth) & set(recommended_ranked_list_top)
                                # print('common_elements in first commit', len(common_elements))
                                # print('common_elements in first commit', common_elements)

                                # precision = len(common_elements) / len(recommended_ranked_list_top)
                                precision = len(common_elements) / top
                                recall = len(common_elements) / len(truth)

                                if (precision + recall) > 0:
                                    f_score = 2 * (precision * recall) / (precision + recall)
                                else:
                                    f_score = 0

                                Common_Elements.append((len(common_elements)))
                                PRECISION_File.append(precision)
                                RECALL_File.append(recall)
                                FSCORE_File.append(f_score)
                                if precision > 0:
                                    SUCESSRATE_File.append(1)
                                else:
                                    SUCESSRATE_File.append(0)

                                # scores per commit of one file ######################
                                #####################

                                print("Precision:", precision)
                                print("Recall:", recall)
                                print("f_score:", f_score)

                                # print(recommended_ranked_list_top)
                                # print(truth)
                                mrr = compute_MRR([recommended_ranked_list_top], [truth])
                                # print(mrr)
                                if math.isnan(mrr):
                                    mrr = 0
                                    print("MRR:", mrr)
                                else:
                                    print("MRR:", mrr)

                                MRR_File.append(mrr)

                                _31_precision_commit.append(precision)
                                _31_recall_commit.append(recall)
                                _31_Fscore_commit.append(f_score)
                                _31_MRR_commit.append(mrr)

                                if precision > 0:
                                    _31_SuccessScore_commit.append(1)
                                else:
                                    _31_SuccessScore_commit.append(0)

                                # print(_31_precision_commit)
                                # print(best_ind)
                                _solution.append(best_ind)

                            _31_best_solution.append(_solution)
                            _31_precision.append(_31_precision_commit)
                            _31_recall.append(_31_recall_commit)
                            _31_Fscore.append(_31_Fscore_commit)
                            _31_MRR.append(_31_MRR_commit)
                            _31_SuccessScore.append(_31_SuccessScore_commit)
                            _31_Co_modified_files.append(_31_Co_modified_files_commit)
                            _31_Nb_co_modified_files.append(_31_Nb_co_modified_files_commit)
                            _31_Recommended_files.append(_31_Recommended_files_commit)
                            _31_IaC_file.append(_31_IaC_file_commit)
                            _31_Nb_recommended_files.append(_31_Nb_recommended_files_commit)
                            _31_commits_hash.append(selected_commits_hash)
                            # print(_31_precision)

                            

                            Tops_precision.append(_31_precision)
                            # print(Tops_precision)
                            Tops_recall.append(_31_recall)
                            Tops_fscore.append(_31_Fscore)
                            Tops_MRR.append(_31_MRR)
                            Tops_SR.append(_31_SuccessScore)
                            # print(_31_best_solution)
                            Tops_BestSolution.append(_31_best_solution)
                            Tops_comodified_files.append(_31_Co_modified_files)
                            Tops_nb_comodified_files.append(_31_Nb_co_modified_files)
                            Tops_recommended_files.append(_31_Recommended_files)
                            Tops_iac_file.append(_31_IaC_file)
                            Tops_nb_recommended_files.append(_31_Nb_recommended_files)
                            Tops_commit_hash.append(_31_commits_hash)

                            # print(Tops_precision)
                            # print(Tops_BestSolution)
                            # print(Tops_iac_file)
                            # print(Tops_commit_hash)
                            # print(Tops_nb_recommended_files)

                        _31_Tops_precision.append(Tops_precision)
                        _31_Tops_recall.append(Tops_recall)
                        _31_Tops_fscore.append(Tops_fscore)
                        _31_Tops_MRR.append(Tops_MRR)
                        _31_Tops_SR.append(Tops_SR)
                        _31_Tops_BestSolution.append(Tops_BestSolution)
                        _31_Tops_comodified_files.append(Tops_comodified_files)
                        _31_Tops_nb_comodified_files.append(Tops_nb_comodified_files)
                        _31_Tops_recommended_files.append(Tops_recommended_files)
                        _31_Tops_iac_file.append(Tops_iac_file)
                        _31_Tops_nb_recommended_files.append(Tops_nb_recommended_files)
                        _31_Tops_commit_hash.append(Tops_commit_hash)

                    # print(_31_Tops_precision)
                    # print(_31_Tops_iac_file)
                    # print(_31_Tops_BestSolution)
                    # print(_31_Tops_nb_recommended_files)
                    # print(_31_Tops_commit_hash)

                    median_indexes = ['Precision_based', 'Recall_based', 'MRR_based',
                                      'Fscore_based', 'Success_Rate_based']

                    for i in range(K):
                        # print(i)

                        precision_top = []
                        for lst in _31_Tops_precision:
                            precision_top.append(lst[i][0])

                        # print(precision_top)
                        # print(len(precision_top))

                        recall_top = []
                        for lst in _31_Tops_recall:
                            recall_top.append(lst[i][0])

                        fscore_top = []

                        for lst in _31_Tops_fscore:
                            fscore_top.append(lst[i][0])

                        mrr_top = []
                        for lst in _31_Tops_MRR:
                            mrr_top.append(lst[i][0])

                        SR_top = []
                        for lst in _31_Tops_SR:
                            SR_top.append(lst[i][0])

                        BestSolution_top = []
                        for lst in _31_Tops_BestSolution:
                            BestSolution_top.append(lst[i][0])

                        comodified_files_top = []
                        for lst in _31_Tops_comodified_files:
                            comodified_files_top.append(lst[i][0])

                        nb_comodified_files_top = []
                        for lst in _31_Tops_nb_comodified_files:
                            nb_comodified_files_top.append(lst[i][0])

                        recommended_files_top = []
                        for lst in _31_Tops_recommended_files:
                            recommended_files_top.append(lst[i][0])

                        # print(recommended_files_top)
                        # print(len(recommended_files_top))

                        iac_file_top = []
                        for lst in _31_Tops_iac_file:
                            iac_file_top.append(lst[i][0])

                        nb_recommended_files_top = []
                        for lst in _31_Tops_nb_recommended_files:
                            nb_recommended_files_top.append(lst[i][0])

                        commit_hash_top = []
                        for lst in _31_Tops_commit_hash:
                            commit_hash_top.append(lst[i][0])

                        precision_median_values, precision_median_indexes = find_median_indexes(precision_top)
                        recall_median_values, recall_median_indexes = find_median_indexes(recall_top)
                        fscore_median_values, fscore_median_indexes = find_median_indexes(fscore_top)
                        MRR_median_values, MRR_median_indexes = find_median_indexes(mrr_top)
                        SR_median_values, SR_median_indexes = find_median_indexes(SR_top)

                        #print(precision_median_indexes)
                        #print(recall_median_indexes)
                        #print(fscore_median_indexes)
                        #print(MRR_median_indexes)
                        #print(SR_median_indexes)

                        precision_average_values = find_average_indexes(precision_top)
                        recall_average_values = find_average_indexes(recall_top)
                        fscore_average_values = find_average_indexes(fscore_top)
                        MRR_average_values = find_average_indexes(mrr_top)
                        SR_average_values = find_average_indexes(SR_top)

                        for metric in median_indexes:
                            # print(metric)

                            # Get the corresponding median list based on the metric
                            if metric == 'Precision_based':
                                median_list = precision_median_indexes
                            elif metric == 'Recall_based':
                                median_list = recall_median_indexes
                            elif metric == 'MRR_based':
                                median_list = MRR_median_indexes
                            elif metric == 'Fscore_based':
                                median_list = fscore_median_indexes
                            elif metric == 'Success_Rate_based':
                                median_list = SR_median_indexes
                            else:
                                median_list = []

                                ### TAKE MEDIAN BASED ON METRIC

                            precision_top_f = [precision_top[i][index] for index, i in enumerate(median_list)]

                            recall_top_f = [recall_top[i][index] for index, i in enumerate(median_list)]
                            fscore_top_f = [fscore_top[i][index] for index, i in enumerate(median_list)]
                            mrr_top_f = [mrr_top[i][index] for index, i in enumerate(median_list)]
                            SR_top_f = [SR_top[i][index] for index, i in enumerate(median_list)]
                            #print(median_list)
                            #print(BestSolution_top)
                            BestSolution_top_f = [BestSolution_top[i][index] for index, i in enumerate(median_list)]

                            comodified_files_top_f = [comodified_files_top[i][index] for index, i in
                                                      enumerate(median_list)]
                            nb_comodified_files_top_f = [nb_comodified_files_top[i][index] for index, i in
                                                         enumerate(median_list)]
                            recommended_files_top_f = [recommended_files_top[i][index] for index, i in
                                                       enumerate(median_list)]
                            iac_file_top_f = [iac_file_top[i][index] for index, i in enumerate(median_list)]
                            nb_recommended_files_top_f = [nb_recommended_files_top[i][index] for index, i in
                                                          enumerate(median_list)]
                            commit_hash_top_f = [commit_hash_top[i][index] for index, i in enumerate(median_list)]

                            # print(commit_hash_top_f)

                            filenameFile = f'{filenameX}.csv'
                            #print("""hereeeee""")
                            #print(BestSolution_top_f)
                            data = {'IaC_file': iac_file_top_f,
                                    'Nb_recommended_files': nb_recommended_files_top_f,
                                    'Best_Individual': BestSolution_top_f,
                                    'Recommended_files': recommended_files_top_f,
                                    'Co_modified_files': comodified_files_top_f,
                                    'Commits_hash': commit_hash_top_f,
                                    'PRECISION_median': precision_top_f,
                                    'RECALL_median': recall_top_f,
                                    'FSCORE_median': fscore_top_f,
                                    'SUCCESS_RATE': SR_top_f,
                                    'MRR_median': mrr_top_f,
                                    'PRECISION_average': precision_average_values,
                                    'RECALL_average': recall_average_values,
                                    'FSCORE_average': fscore_average_values,
                                    'MRR_average': MRR_average_values}
                            df = pd.DataFrame(data)

                            df.to_csv(
                                Project_path + project + '/' + Algo + '/' + str(
                                    tops[i]) + '/' + metric + '/' + Results_Median + '/' + filenameFile,
                                index=False)

                            filenameFile = f'{filenameX}.csv'
                            data = {'NB_recommended_files': nb_recommended_files_top,
                                    'Best_solution': BestSolution_top,
                                    'Recommended_files': recommended_files_top,
                                    'Co_modified_files': comodified_files_top,
                                    'Commits_hash': commit_hash_top,
                                    'PRECISION': precision_top, 'RECALL': recall_top,
                                    'FSCORE': fscore_top, 'SUCCESS_RATE': SR_top,
                                    'MRR': mrr_top}
                            df = pd.DataFrame(data)

                            df.to_csv(
                                Project_path + project + '/' + Algo + '/' + str(
                                    tops[i]) + '/' + metric + '/' + _31_runs + '/' + filenameFile,
                                index=False)

        for i in range(K):
            # print('top', tops[i])

            for metric in median_indexes:
                # print(len(all_commits_hash))

                for Interest_hash in all_commits_hash:

                    # print(Interest_hash)

                    Precision_commit = []
                    Recall_commit = []
                    Fscore_commit = []
                    SR_commit = []
                    MRR_commit = []
                    file_commit = []
                    file_commit_list = []
                    Co_modified_files_commit_ = []

                    Parent_folder_path = Project_path + project + '/' + Algo + '/' + str(tops[i]) + '/' + metric
                    for folder_name in os.listdir(Parent_folder_path):
                        # print(folder_name)

                        # Check if the folder name contains the string "FilesPlots"
                        if 'Results_Median' in folder_name and os.path.isdir(Parent_folder_path):
                            # print("yes")
                            folder_path = os.path.join(Parent_folder_path, folder_name)
                            # print(folder_path)
                            for filename in os.listdir(folder_path):
                                if filename.endswith('.csv'):
                                    # print(filename)
                                    file_path = os.path.join(folder_path, filename)
                                    # print(file_path)
                                    with open(file_path, 'r') as csv_file:
                                        reader = csv.DictReader(csv_file)

                                        for row in reader:

                                            # print(row)
                                            try:
                                                hash = row['Commits_hash']
                                                # file = row['IaC_file']
                                                #print(hash)
                                                if hash == Interest_hash:
                                                    #print("yes", hash)
                                                    #  print(Interest_hash)

                                                    file_commit.append(row['IaC_file'])

                                                    Precision_commit.append(float(row['PRECISION_median']))
                                                    # print(Precision_commit)
                                                    Recall_commit.append(float(row['RECALL_median']))
                                                    Fscore_commit.append(float(row['FSCORE_median']))
                                                    SR_commit.append(float(row['SUCCESS_RATE']))
                                                    MRR_commit.append(float(row['MRR_median']))

                                                    Co_modified_files_commit_.append(row['Co_modified_files'])
                                                    # print(Co_modified_files_commit_)
                                                    for file_ in Co_modified_files_commit_:
                                                        # print(file_)

                                                        result_list = eval(file_)
                                                        # print(result_list)

                                                        for f in result_list:

                                                            if '(Other)' in f:
                                                                # print('yes')
                                                                if f not in file_commit:
                                                                    file_commit.append(f)
                                                                    Precision_commit.append(float(0.0))
                                                                    Recall_commit.append(float(0.0))
                                                                    Fscore_commit.append(float(0.0))
                                                                    SR_commit.append(float(0.0))
                                                                    MRR_commit.append(float(0.0))

                                            except (ValueError, KeyError):
                                                continue

                    filenamehash = f'{Interest_hash}_.csv'
                    output_file_path = Project_path + project + '/' + Algo + '/' + str(
                        tops[i]) + '/' + metric + '/' + Results_Commit_level + '/' + filenamehash

                    file_commit_list = file_commit

                    if len(Precision_commit) > 0:
                        with open(output_file_path, 'w', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow(
                                ['IaC_File', 'Precision', 'Recall', 'Fscore', 'Success_rate', 'MRR'])
                            for k in range(len(Precision_commit)):
                                writer.writerow(
                                    [file_commit_list[k], Precision_commit[k], Recall_commit[k],
                                     Fscore_commit[k],
                                     SR_commit[k], MRR_commit[k]])

                        print("Values written to", output_file_path)

        