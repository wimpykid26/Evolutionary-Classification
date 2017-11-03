from dna import dna
from numpy import interp
class Population(p, m, num):
    def __init__(self, p, m, num):
        self.population = []
        self.mating_pool = []
        self.target = p
        self.generations = 0
        self.finished = False
        self.mutation_rate = m
        self.perfect_score = 1
        self.best = ''

        for i in range(0, num):
            self.population.append(dna(len(self.target))
        
        self.calc_fitness()

    
    def calc_fitness(self):
        for i in range(0, len(self.population)):
            self.population[i].calc_fitness(target)
    
    def natural_selection(self):
        max_fitness = 0
        for i in range(0, len(self.population)):
            if self.population[i].fitness > max_fitness:
                max_fitness = self.population[i].fitness

        for i in range(0, len(self.population)):
            fitness = interp(self.population[i].fitness, [0, max_fitness], [0, 1])
            n = floor(fitness * 100)
            for j in range(0, n):
                self.mating_pool.append(self.population[i])

        
    
    
        
