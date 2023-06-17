import os
import time
import folium
import datetime
import numpy as np
import pandas as pd 
from geopy.geocoders import Nominatim


class EvoPoolOpt:
    """
    Class that implements a set of function to optimize the distribution of cities inside sport pools 
    with an simple evolutionnary algorithm using NumPy.
    :param cities_list: (List['str']) List containing the cities names / adress
    :param nb_pools: (int) number of pools in which we want to distribute the cities 
    :param nb_chromosomes: (int) number of chromosomes used in the evolutionnary algorithm
    """
    def __init__(self, cities_list, nb_pools=6, nb_chromosomes=100):
        # Initialization attributes
        self.cities_list = cities_list
        self.cities_df = self.create_cities_df(cities_list)
        self.nb_pools = nb_pools
        self.nb_cities = len(cities_list)
        self.nb_chromosomes = nb_chromosomes
        self.distance_matrix = self.calculate_distance_matrix()
        self.pairwise_distances = np.triu(self.distance_matrix, k=1)
        self.check_duplicates()


    """ Initialization functions """

    def initialize_population(self):
        """
        Initialization of a chromosome population. Each chromosome contains is a np array row with the pool associated to each city index 
        :return population: np.array((self.nb_chromosomes, self.nb_cities)) 
        """
        population = np.zeros((self.nb_chromosomes, self.nb_cities), dtype=np.int64)
        

        for chr in range(self.nb_chromosomes):
            # At the creation of the chromosome, each pools are empty
            non_full_pools = [pool for pool in range(self.nb_pools)]
            # We keep track of the number of cities per pool with this list
            city_per_pool = [ 0 for pool in range(self.nb_pools)]
            # We iterate through the DataFrame
            for idx, city in enumerate(self.cities_df['city']): 
                # We match the city index with a pool that isn't full yet
                pool_choice = np.random.choice(non_full_pools)
                population[chr, idx] = pool_choice
                city_per_pool[pool_choice] += 1
                if city_per_pool[pool_choice] >= 12 :  
                    non_full_pools.remove(pool_choice) 

        return population
    

    def create_cities_df(self, cities_list):
        """
        Creation (or Loading) of a DataFrame containing the cities names and their associated coordinates
        :param cities_list: (List['str']) List containing the cities names / adress
        :return cities_df: pd.DataFrame
        """
        
        filename = 'df_cities.csv'

        # Load file if it already exists
        if os.path.isfile(filename):
            print(f"Downloading existing {filename}\n")
            cities_df = pd.read_csv(filename)
        # Else create it 
        else:
            cities_df= pd.DataFrame(cities_list, columns=['city'])
            
            start = time.time()
            print(f"Downloading cities GPS coordinates ...", end='')
            geolocator = Nominatim(timeout=10, user_agent = "myGeolocator")
            cities_df['gcode'] = cities_df['city'].apply(geolocator.geocode)
            print(f"\rcities GPS coordinates downloaded in {time.time() - start:.1f} seconds", end='')

            cities_df['lat'] = [g.latitude for g in cities_df.gcode]
            cities_df['long'] = [g.longitude for g in cities_df.gcode]

            # Save it to make it available for another time
            cities_df.to_csv(filename, index=False)
        
        return cities_df
    

    def check_duplicates(self):
        """
        Check wether self.cities_list contains duplicates and print them if it is the case
        """
        if self.cities_list != set(self.cities_list):
            seen_before = {}
            for city in self.cities_list:
                if city in seen_before:
                    seen_before[city] += 1
                else:
                    seen_before[city] = 1

            duplicates = [city for city, count in seen_before.items() if count > 1]
            print("The following cities are duplicated: ")
            for city in duplicates:
                print(city)
        else:
            pass
    

    """ Distance calculations functions """

    def calculate_distance_matrix(self):
        """
        Calculates the distance between all the cities and store it inside a matrix 
        :return self.dist_matrix: (np.array) 
        """
        latitudes = np.array(self.cities_df['lat'])
        longitudes = np.array(self.cities_df['long'])

        lat_diff = latitudes[:, np.newaxis] - latitudes
        lon_diff = longitudes[:, np.newaxis] - longitudes
        dist_matrix = np.sqrt(lat_diff**2 + lon_diff**2)

        return dist_matrix
    
    
    def calculate_chromosome_distance(self, chromosome, pools_details=False):
        """
        Calculate the distance travelled inside each pool of a chromosome and returns the distance sum across
        all the pools. If verbose == True, prints the detail of the distance within each pool.
        :param chromosome: (np.array) Chromosome array 
        :param pools_details: (bool) Wether to print the distances details or not 
        :return chromosome_distance: (float) Sum of distances travelled across all pools 
        """
        chr_pool_distances = np.zeros(self.nb_pools, dtype=np.int64)
        # Loop over each pool number
        for pool in range(self.nb_pools):
            pool_indices = np.where(chromosome == pool)[0]
            pool_distance = self.pairwise_distances[np.ix_(pool_indices, pool_indices)]
            chr_pool_distances[pool] = np.sum(pool_distance)
        
        if pools_details: 
            print("Distance within each pool : \n")
            print({pool: chr_pool_distances[pool] for pool in range(self.nb_pools)})
        return np.sum(chr_pool_distances)
    

    def calculate_population_distance(self, population):
        """
        Apply the calculate_chromosome_distance to the whole population
        :param population: (np.array) Population array 
        :return population_distance: (np.array) array of distances for each chromosome of the population
        """
        return np.apply_along_axis(self.calculate_chromosome_distance, axis=1, arr=population)
    

    def get_fitness_statistics(self, population):
        """
        Returns statistics on a population distance array 
        :param population: (np.array) Population array 
        :return mean, min, max: (float, float, float) mean, min and max of the population distances
        """
        population_distances = self.calculate_population_distance(population)
        mean = np.mean(population_distances)
        min = np.min(population_distances)
        max = np.max(population_distances)
        return mean, min, max
    

    def order_fitness_list(self, population_distances):
        """
        Order the population fitness_list (=population distances)
        :param population_distances: (np.array) Population distances array 
        :return ordered_fitness_list: (np.array(self.nb_chromosomes, 2)) Ordrered fitness list with associated indices
        """
        # Creating indices for each chromosome fitness 
        indices = np.arange(self.nb_chromosomes, dtype=int)
        population_distance_w_idx = np.column_stack((population_distances, indices))
        # Ordering the fitnesses 
        ordered_indices = np.argsort(population_distance_w_idx[:,0])
        ordered_fitness_list = population_distance_w_idx[ordered_indices]
        return ordered_fitness_list
    

    """ New population creation functions """

    def select_best_chromosomes(self, ordered_fitness_list, population, nb_chromosomes=20):
        """
        Select the best chromosomes of a population
        :param ordered_fitness_list: (np.array) 
        :param population: (np.array) 
        :param nb_chromosomes: (int) Number of chromosomes to be selected
        :return best_chromosomes: (np.array) Best chromosomes selected
        """
        best_chr_idx = ordered_fitness_list[:nb_chromosomes, 1]
        best_chromosomes = population[best_chr_idx]
        return best_chromosomes
    
    # On calcule la proba pour chaque chromosome d'être parent par rapport à son rang dans la fitness_list:
    def calculate_parent_proba(self, ordered_fitness_list):
        """
        Calculate the probability of chromosomes to be a parent based on their ranks
        :param ordered_fitness_list: (np.array) 
        :return parents_probabilities: (np.array) probability of each chromosome of the population to be a parent
        """
        ranks = np.arange(self.nb_chromosomes)
        probabilities = self.nb_chromosomes - ranks
        probabilities = probabilities / np.sum(probabilities)

        # probabilities but for sorted chromosomes : 
        ordered_fitness_chromosome_idx = ordered_fitness_list[:, 1]
        parents_probabilities = np.zeros((probabilities.shape))
        parents_probabilities[ordered_fitness_chromosome_idx] = probabilities

        return parents_probabilities


    def select_parents(self, parents_probabilities, population, nb_parents=80):
        """
        Select the parents based on the probabilities calculated with precedent function
        :param parents_probabilities: (np.array) 
        :param population: (np.array) 
        :param nb_parents: (int) Number of parents selected 
        :return parents_probabilities: (np.array) probability of each chromosome of the population to be a parent
        """
        selected_parents_idx = np.random.choice(np.arange(self.nb_chromosomes), p=parents_probabilities, size=nb_parents)
        selected_parents = population[selected_parents_idx]
        
        return selected_parents
    
    def mutate_parent_chromosome(self, chromosome, num_permutations):
        """
        Mutate a chromosome by perumting city pools
        :param chromosome: (np.array) Parent chromosome selected
        :param num_permutations: (int) Number of permutations to make
        :return mutated_chromosome: (np.array) Mutated chromosome
        """
        # We reshape the permutations idx to get the cities permuted in the same row
        permutation_idx = np.random.choice(np.arange(self.nb_cities), size=num_permutations).reshape(-1,2)

        mutated_chromosome = chromosome.copy()

        mutated_chromosome[permutation_idx[:,0]], mutated_chromosome[permutation_idx[:,1]] = \
        mutated_chromosome[permutation_idx[:,1]], mutated_chromosome[permutation_idx[:,0]]
        
        return mutated_chromosome
    

    def mutate_parent_population(self, parent_population, num_permutations):
        """
        Apply mutate_parent_chromosome on the selected parent population
        :param parent_population: (np.array) Parent population selected
        :param num_permutations: (int) Number of permutations to make
        :return mutated_chromosome: (np.array) Mutated chromosome
        """
        return np.apply_along_axis(self.mutate_parent_chromosome, axis=1, arr=parent_population, num_permutations=num_permutations)
    

    def create_new_population(self, best_chromosomes, mutated_chromosomes):
        """
        Concatenates best and mutated chromosomes from last population to create a new one 
        :param best_chromosomes: (np.array)
        :param mutated_chromosomes: (np.array)
        :return population: (np.array) New population 
        """
        return np.concatenate((best_chromosomes, mutated_chromosomes), axis=0)
    

    """ Optimization functions """

    def optimize(self, num_iterations=50_000, measurement_step=100):
        """
        Launch an optimization with the evolutionnary algorithm and returns the best population found 
        :param num_iterations: (int) number of algorithm iterations
        :param measurement_step: (int) measurement and plotting step of the performance
        :return best_population: (np.array) Best population found
        :return ordered_fitness_list: (np.array) Ordered_fitness_list of best population found
        :return min_distance: (np.array) Evolution of min distance through time
        """
        nb_measures = num_iterations//measurement_step
        measure_idx = 0
        
        self.mean_distance = np.zeros((nb_measures,))
        self.min_distance = np.zeros((nb_measures,))
        self.max_distance = np.zeros((nb_measures,))

        start = time.time()

        # define an initial population
        population = self.initialize_population()

        for i in range(1, num_iterations+1): 
            
            # calculate the fitness of a population and order its chromosomes fitness by rank
            population_distances = self.calculate_population_distance(population)
            ordered_fitness_list = self.order_fitness_list(population_distances)

            # log and plot the results every 'measurement_steps'
            if i % measurement_step == 0:
                mean, min, max = self.get_fitness_statistics(population)
                self.mean_distance[measure_idx] = mean
                self.min_distance[measure_idx] = min
                self.max_distance[measure_idx] = max
                measure_idx += 1 
                print(f"\rIteration {i} statistics : mean : {mean:.2f}  min : {min:.2f}  max : {max:.2f}", end='')

            # mutate chromosomes from the last population to create a new subset of the population
            ordered_probabilities = self.calculate_parent_proba(ordered_fitness_list)
            parents = self.select_parents(parents_probabilities=ordered_probabilities, population=population, nb_parents=80)
            # between 2 and 10 permutations
            num_permutations = 2 * np.random.randint(1, 5) 
            mutated_chromosomes = self.mutate_parent_population(parents, num_permutations)
            # select some of the best chromosomes of the last population 
            best_chromosomes = self.select_best_chromosomes(ordered_fitness_list=ordered_fitness_list, population=population, nb_chromosomes=20)

            # add the best and the mutated chromosomes to create the new population
            population = self.create_new_population(best_chromosomes, mutated_chromosomes=mutated_chromosomes) 

        best_chromosome_idx = ordered_fitness_list[0, 1]
        self.best_chromosome = population[best_chromosome_idx]

        end = time.time()
        print(f"\nOptimization done in : {str(datetime.timedelta(seconds = end-start))}")


    """ Transformation functions """

    def dict_to_chromosome(self, dict):
        """
        Transform a dictionnary of handwritted pool configuration to chromosome
        :param dict: (dict[pools]) Pools configuration in a dictionnary 
        :return chromosome: (np.array) Pools configuration in a chromosome
        """
        chromosome = np.zeros(self.nb_cities, dtype=np.int32)
        for pool_idx, pool in enumerate(dict):
            for city in dict[pool]:
                city_idx = self.cities_list.index(city)
                chromosome[city_idx] = pool_idx
        return chromosome


    """ Display functions """
    
    def get_dict_chromosome(self, chromosome):
        """
        Returns a dictionnary with {city: pool} for a chromosome 
        :param chromosome: (np.array) 
        :return dict: (dict[pools])
        """
        return {city: pool for city, pool in zip(self.cities_list, chromosome)}
         

    def show_population(self):
        """
        Shows the population shape and the first five chromosomes 
        """
        print(f"population shape : {np.asarray(self.population).shape}")
        for chr_idx in range(5):
            print(self.display_dict_chromosome(self.population[chr_idx, :]))


    def display_cities_map(self):
        """ 
        Display the cities of the cities_df on a map
        """
        map = folium.Map(location=(47.0000,2.0000), zoom_start=6)
        
        for index, row in self.cities_df.iterrows():
            folium.Marker(location=(row['lat'], row['long'])).add_to(map)
        display(map)

    
    def display_map_pools_configuration(self, chromosome):
        """ 
        Display the pools configuration of a chromosome on a map
        """
        map = folium.Map(location=(47.0000,2.0000), zoom_start=6)
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred']
        # Display the cities with the color corresponding to their pool 
        for city_idx, city_pool in enumerate(chromosome):
            latitude, longitude = self.cities_df.iloc[city_idx,2], self.cities_df.iloc[city_idx,3]
            folium.Marker(location=(latitude, longitude), icon=folium.Icon(color=colors[city_pool])).add_to(map)

        display(map)