__author__ = 'GongLi'

from pulp import *
import numpy as np

def EMD(feature1, feature2, w1, w2):
    os.environ['PATH'] += os.pathsep + '/usr/local/bin'

    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)


    tempMin = min(sum(w1), sum(w2))
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) <= w1[i]

    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) <= w2[j]

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(GLPK_CMD())

    flow = value(problem.objective)


    return flow / tempMin


if __name__ == '__main__':
    feature1 = np.array([[100, 40, 22], [211,20,2], [32, 190, 150], [ 2, 100, 100]])
    feature2 = np.array([[0,0,0], [50, 100, 80], [255, 255, 255]])

    w1 = [0.4,0.3,0.2,0.1]
    w2 = [0.5, 0.3, 0.2]


    emdDistance = EMD(feature1, feature2, w1, w2)
    print str(emdDistance)




           





