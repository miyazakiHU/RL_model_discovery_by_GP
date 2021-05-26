import operator
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from utils import get_act_reward
from funcs import eval_func
act1, reward1 = get_act_reward()

# Define new functions
def protectedDiv(left, right):
    if (right != 0):
        return left / right
    else:
        return 1

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
# pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
constant_list = [-1, -0.5, 0, 0.5, 1]
pset.addEphemeralConstant("rand101", lambda: random.choice(constant_list))
# pset.renameArguments(ARG0='x')
pset.renameArguments(ARG0='q')
pset.renameArguments(ARG1='r')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 評価関数
def evalSymbReg(individual, penalty=0.5):
    func = toolbox.compile(expr=individual)
    loglikelihood = eval_func(func, 0.3, act1, reward1) + penalty*len(individual)
    # print(f"The length of {individual} is :", len(individual))
    return loglikelihood,


toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(1)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats,
                               halloffame=hof, verbose=True)

expr = tools.selBest(pop, 1)[0]
tree = gp.PrimitiveTree(expr)
print(str(tree))

# Output the result
with open("output\\result.txt", mode='w') as f:
    f.writelines(str(tree))