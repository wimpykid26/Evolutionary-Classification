from dna import dna
from numpy import interp
from math import floor
from random import randint


class Population():

    def calc_fitness_population(self):
        for i in range(0, len(self.population)):
            self.population[i].calc_fitness(self.target)

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
            self.population.append(dna(len(self.target)))
        
        self.calc_fitness_population()
    
    def natural_selection(self):
        max_fitness = 0
        for i in range(0, len(self.population)):
            if self.population[i].fitness > max_fitness:
                max_fitness = self.population[i].fitness

        for i in range(0, len(self.population)):
            fitness = interp(self.population[i].fitness, [0, max_fitness], [0, 1])
            n = int(floor(fitness * 100))
            for j in range(0, n):
                self.mating_pool.append(self.population[i])

        
    def generate(self):
        for i in range(0, len(self.population)):
            a = int(floor(randint(0, len(self.mating_pool))))
            b = int(floor(randint(0, len(self.mating_pool))))
            partner_a = self.mating_pool[a]
            partner_b = self.mating_pool[b]
            child = partner_a.crossover(partner_b)
            child.mutate(self.mutation_rate)
            self.population[i] = child
        
        self.generations = self.generations + 1
    
    def get_best(self):
        return self.best

    def evaluate(self):
        world_record = 0.0
        index = 0
        for i in range(0, len(self.population)):
            if self.population[i].fitness > world_record:
                world_record = self.population[i].fitness
                index = i
            
        self.best = self.population[i].fitness
        if world_record == self.perfect_score:
            self.finished = true

    
    def is_finished(self):
        return self.finished

    def get_generations(self):
        return self.get_generations

    def get_average_fitness(self):
        total = 0
        for i in range(0, len(self.population)):
            total += self.population[i].fitness
        
        return floor(float(total/len(self.population)))

    def all_phrases(self):
        everything = ''
        display_limit = min(len(self.population), 50)
        for i in range(0, display_limit):
            everything += self.population[i].get_phrase() + "\n"

        return everything
