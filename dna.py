from random import randint
from math import floor

class dna(num):
    
    def newChar(self):
        c = floor(randint(63, 122))
        if c is 63:
            c = 32
        if c is 64:
            c = 46    
        return chr(c)

    def __init__(num=10):
        self.num = num
        self.genes = []
        self.fitness = 0
        for i in range (0, num):
            self.genes.append(self.newChar())
        
    def getPhrase(self):
        return ''.join(this.genes)

    def calcFitness(self, target):
        score = 0
        for i in range(0, len(self.genes)):
            if self.genes[i] = target[i]:
                score++
            
        self.fitness = score/len(target)
    
    def crossover(self, partner):
        child = dna(len(self.genes))
        midpoint = floor(randint(len(self.genes)))

        for i in range(0, len(self.genes)):
            
