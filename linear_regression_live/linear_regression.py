from numpy import *
import numpy as np
import matplotlib.pyplot as plt


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def dataset_input_diameter(crosssection_name):
    points = genfromtxt(crosssection_name, delimiter=",")
    learning_rate = 0.0001
    initial_b = 15  # initial y-intercept guess
    initial_m = 0.1  # initial slope guess
    num_iterations = 10000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                                 compute_error_for_line_given_points(
                                                                                     initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                        compute_error_for_line_given_points(b, m,
                                                                                                            points)))
    K = np.linspace(0, 100, 100)
    C = [m * k + b for k in K]
    plt.scatter(points[0:100, 0], points[0:100, 1])
    plt.plot(K, C, color='r')
    plt.show()
    return [b, m]


def dataset_input_curvilinear(crosssection_name):
    points = genfromtxt(crosssection_name, delimiter=",")
    learning_rate = 0.0001
    initial_b = 15  # initial y-intercept guess
    initial_m = 0.1  # initial slope guess
    num_iterations = 10000
    # print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
    #                                                                              compute_error_for_line_given_points(
    #                                                                                  initial_b, initial_m, points)))
    # print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    # print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
    #                                                                      compute_error_for_line_given_points(b, m,
    #                                                                                                          points)))
    # K = np.linspace(0, 100, 100)
    # C = [m * k + b for k in K]
    # plt.scatter(points[0:100, 0], points[0:100, 1])
    # plt.plot(K, C, color='r')
    # plt.show()
    return [b, m]


def dataset_input_HandW(crosssection_name):
    points = genfromtxt(crosssection_name, delimiter=",")
    learning_rate = 0.0001
    initial_b = 20  # initial y-intercept guess
    initial_m = 0.05  # initial slope guess
    num_iterations = 10000
    # print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
    #                                                                              compute_error_for_line_given_points(
    #                                                                                  initial_b, initial_m, points)))
    # print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    # print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
    #                                                                      compute_error_for_line_given_points(b, m,
    #                                                                                                          points)))
    # K = np.linspace(0, 100, 100)
    # C = [m * k + b for k in K]
    # plt.scatter(points[0:100, 0], points[0:100, 1])
    # plt.plot(K, C, color='r')
    # plt.show()
    return [b, m]


def dataset_input_curvilinear1_11(crosssection_name):
    points = genfromtxt(crosssection_name, delimiter=",")
    learning_rate = 0.0001
    initial_b = 200  # initial y-intercept guess
    initial_m = 0.1  # initial slope guess
    num_iterations = 10000
    # print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
    #                                                                             compute_error_for_line_given_points(
    #                                                                                 initial_b, initial_m, points)))
    # print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    # print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
    #                                                                     compute_error_for_line_given_points(b, m,
    #                                                                                                         points)))
    # K = np.linspace(0, 100, 100)
    # C = [m * k + b for k in K]
    # plt.scatter(points[0:100, 0], points[0:100, 1])
    # plt.plot(K, C, color='r')
    # plt.show()
    return [b, m]


def run():
    # mean diameter
    [b, m] = dataset_input_diameter("diameter_cs1.csv")
    [b0, m0] = dataset_input_diameter("diameter_cs2.csv")
    [b1, m1] = dataset_input_diameter("diameter_cs3_cs5.csv")
    [b2, m2] = dataset_input_diameter("diameter_cs5_cs8.csv")
    [b3, m3] = dataset_input_diameter("diameter_cs8_cs11.csv")

    # curvilinear length
    [b4, m4] = dataset_input_curvilinear("curvilinear length cs1-5.csv")
    [b5, m5] = dataset_input_curvilinear1_11("curvilinear lengthcs1-11.csv")
    [b6, m6] = dataset_input_curvilinear("curvilinear lengthCS3-5.csv")
    [b7, m7] = dataset_input_curvilinear("curvilinear lengthCS5-8.csv")
    [b8, m8] = dataset_input_curvilinear("curvilinear lengthCS8-11.csv")

    # height
    [b9, m9] = dataset_input_HandW("height.csv")

    # width
    [b10, m10] = dataset_input_HandW("width.csv")

    again = 'Y'
    # results = []
    while again == 'Y':

        age = int(input(print('enter the age:')))
        cs1 = m * age + b
        cs2 = m0 * age + b0
        cs3_cs5 = m1 * age + b1
        cs5_cs8 = m2 * age + b2
        cs8_cs11 = m3 * age + b3

        Ccs1_cs5 = m4 * age + b4
        Ccs1_cs11 = m5 * age + b5
        Ccs3_cs5 = m6 * age + b6
        Ccs5_cs8 = m7 * age + b7
        Ccs8_cs11 = m8 * age + b8

        height = m9 * age + b9
        width = m10 * age + b10

        print(m)
        print(b)
        print(m0)
        print(b0)
        print(m1)
        print(b1)
        print(m2)
        print(b2)
        print(m3)
        print(b3)
        print('m4')
        print(m4)
        print(b4)
        print(m5)
        print(b5)
        print(m6)
        print(b6)
        print(m7)
        print(b7)
        print(m8)
        print(b8)
        print(m9)
        print(b9)
        print(m10)
        print(b10)

        # Results.extend([cs1, cs2, cs3_cs5, cs5_cs8, cs8_cs11, Ccs3_cs5, Ccs5_cs8, Ccs8_cs11, Ccs1_cs11, height, width])
        # print(Results)
        # Results = []
        print('\ncs1mean diameter is ' + '{}'.format(cs1))
        print('\ncs2mean diameter is ' + '{}'.format(cs2))
        print('\ncs3-cs5mean diameter is ' + '{}'.format(cs3_cs5))
        print('\ncs5-cs8 mean diameter is ' + '{}'.format(cs5_cs8))
        print('\ncs8-cs11mean diameter is ' + '{}'.format(cs8_cs11))

        print('\ncs1_cs3curvilinear length is ' + '{}'.format(Ccs1_cs5))
        print('\ncs1-cs11curvilinear length is ' + '{}'.format(Ccs1_cs11))
        print('\ncs3-cs5curvilinear length is ' + '{}'.format(Ccs3_cs5))
        print('\ncs5-cs8curvilinear length is ' + '{}'.format(Ccs5_cs8))
        print('\ncs8-cs11curvilinear length is ' + '{}'.format(Ccs8_cs11))

        print('\nHeight is ' + '{}'.format(height))
        print('\nWidth is ' + '{}'.format(width))

        again = input(print('start again?(Y/N):'))
    else:
        print('finish')


if __name__ == '__main__':
    run()
