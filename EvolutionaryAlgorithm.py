import random
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def linear_kernel(X, Z=None):
    if Z is None:
        return np.dot(X, X.T)
    else:
        return np.dot(X, Z.T)

def adjustment_algorithm(alpha_genes, y_train, C):
   
    alpha = np.array(alpha_genes)
    L = len(y_train)
    
    s = np.sum(alpha * y_train)

    
    while s!=0:
    
        s_plus = np.sum(alpha[y_train == 1] * y_train[y_train == 1])
        s_minus = np.sum(alpha[y_train == -1] * y_train[y_train == -1])
        
        
        if s_plus > s_minus:

            indices_plus = np.where(y_train == 1)[0]
            if len(indices_plus) > 0:
                k = random.choice(indices_plus)
        else: 
            indices_minus = np.where(y_train == -1)[0]
            if len(indices_minus) > 0:
                k = random.choice(indices_minus)
        
        
        if alpha[k] > s: 
            
            alpha[k] = alpha[k] - s
       
        else: 
           
            alpha[k] = 0.0

    
        s = np.sum(alpha * y_train)

   
    alpha = np.clip(alpha, 0, C)
    
    return alpha.tolist()


class IOptimizationProblem:
    def compute_fitness(self, chromosome):
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def make_chromosome(self):
        raise NotImplementedError("This method needs to be implemented by a subclass")

class SVM_GATE(IOptimizationProblem):
    def __init__(self, X, y, C, kernel_matrix):
        self.X = X
        self.y = y
        self.C = C
        self.K = kernel_matrix
        self.L = len(y) 
        self.min_vals = [0.0] * self.L #constraint: alpha>=0
        self.max_vals = [self.C] * self.L #contraint: alpha <=C

    def make_chromosome(self):
        chromosome = Chromosome(self.L, self.min_vals, self.max_vals)
        #initial adjustment
        chromosome.genes=adjustment_algorithm(chromosome.genes, self.y, self.C)
        return chromosome

    # F(alpha)=-sum(alpha_i)+1/2*sum_sum(alpha_i*alpha_j*y_i*y_j*<x_i, x_j>), i=1...l, j=1...j
    def compute_fitness(self, chromosome):
        alpha_adjusted = adjustment_algorithm(chromosome.genes, self.y, self.C)
        chromosome.genes = alpha_adjusted
        alpha = np.array(alpha_adjusted)

        F_alpha= -np.sum(alpha) + 0.5* np.dot((alpha * self.y), np.dot(self.K, (alpha*self.y)))
        chromosome_fitness=-F_alpha
    

class Chromosome:
    def __init__(self, no_genes, min_values, max_values):
        self.no_genes = no_genes
        self.genes = [0.0] * no_genes
        self.min_values = list(min_values)
        self.max_values = list(max_values)
        self.fitness = 0.0
        self._initialize_genes()

    def _initialize_genes(self):
        for i in range(self.no_genes):
            self.genes[i] = self.min_values[i] + random.random() * (self.max_values[i] - self.min_values[i])

    def __copy__(self):
        new_copy = Chromosome(self.no_genes, self.min_values, self.max_values)
        new_copy.genes = list(self.genes)
        new_copy.fitness = self.fitness
        return new_copy

    def copy_from(self, other):
        self.no_genes = other.no_genes
        self.genes = list(other.genes)
        self.min_values = list(other.min_values)
        self.max_values = list(other.max_values)
        self.fitness = other.fitness

#elitism selection
class Selection:
    @staticmethod
    def get_best(population):
        best_chromosome = population[0]
        for c in population[1:]:
             if c.fitness > best_chromosome.fitness:
                best_chromosome = c
        return best_chromosome.__copy__()

#arithmetic crossover 
class Crossover:
    @staticmethod
    def arithmetic(mother, father, rate):
        child=mother.__copy__()
        r=random.random()
        if r<rate:
            a=random.random()
            for i in range(child.no_genes):
                child.genes[i]=a*mother.genes[i]+(1.0-a)*father.genes[i]
        return child

#swap mutation
class Mutation:
    @staticmethod
    def swap(child, rate):
        
        if random.random() < rate:
            
            if child.no_genes < 2:
                return 

            idx1, idx2 = random.sample(range(child.no_genes), 2)
            
            #swap the values between the genes
            child.genes[idx1], child.genes[idx2] = child.genes[idx2], child.genes[idx1]


class EvolutionaryAlgorithm:
    def solve(self, problem, population_size, max_generations, crossover_rate, mutation_rate):

        population = [problem.make_chromosome() for _ in range(population_size)]
        for individual in population:
            problem.compute_fitness(individual)

        for gen in range(max_generations):
            new_population = [Selection.get_best(population)]

            for i in range(1, population_size):
                pass
                # elitism selection
                mother = Selection.get_best(population)
                father = Selection.get_best(population)
                
                # arithmetic crossover 
                child = Crossover.arithmetic(mother, father, crossover_rate)
                
                #  swap mutation
                Mutation.swap(child, mutation_rate)

                problem.compute_fitness(child)

                # introducere copil in new_population
                new_population.append(child)

            population = new_population

        return Selection.get_best(population)


if __name__ == "__main__":

    POP_SIZE = 50       
    MAX_GEN = 30       
    C_RATE = 0.9       
    M_RATE = 0.1  

    C=1.0 #maybe tuned with k-fold crossover

    df = pd.read_csv("source/train_data.csv", index_col=0)
    X_train = df.drop(columns=['Category'])
    y_train = df['Category']   
    X_train = X_train.values
    y_train = y_train.values
    #l=len(y_train)   

    X_test= pd.read_csv("source/test_data.csv", index_col=0)
    y_test=pd.read_csv("source/test_y.csv")
    X_test = X_test.values
    y_test = y_test = y_test['Category'].values.ravel()

    K_matrix = linear_kernel(X_train)
    svm_problem = SVM_GATE(X_train, y_train, C, K_matrix)
    ea = EvolutionaryAlgorithm()

    solution = ea.solve(svm_problem, POP_SIZE, MAX_GEN, C_RATE, M_RATE) 
    

   

   