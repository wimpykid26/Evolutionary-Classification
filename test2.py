from population import Population
target = "To be or not to be."
popmax = 200
mutationRate = 0.01
population = Population(target, mutationRate, popmax)


def draw():
    while not population.is_finished():
        population.natural_selection()
        population.generate()
        population.calc_fitness_population()
        population.evaluate()
        display_info()


def display_info():
    answer = population.get_best();
    print("Best Phrase:     ", answer , "\n")
    print("Total Generations:     ", population.get_generations(), "\n")
    print("Average Fitness:     ", population.get_average_fitness(), "\n")

  
draw()