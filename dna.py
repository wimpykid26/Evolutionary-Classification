from random import randint, uniform
from math import floor

class dna():
    
    def new_char(self):
        c = int(floor(randint(63, 122)))
        if c is 63:
            c = 32
        if c is 64:
            c = 46    
        return chr(c)

    def __init__(self, num=10):
        self.num = num
        self.genes = []
        self.fitness = 0
        for i in range (0, num):
            self.genes.append(self.new_char())
        
    def get_phrase(self):
        return ''.join(this.genes)

    def calc_fitness(self, target):
        score = 0
        for i in range(0, len(self.genes)):
            if self.genes[i] == target[i]:
                score = score + 1
            
        self.fitness = score/len(target)
    
    def crossover(self, partner):
        child = dna(len(self.genes))
        midpoint = int(floor(randint(0, len(self.genes))))

        for i in range(0, len(self.genes)):
            if i > midpoint:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = partner.genes[i]

        return child

    def mutate(self, mutation_rate):
        for i in range(0, len(self.genes)):
            if uniform(0, 1) < mutation_rate:
                self.genes[i] = self.new_char()