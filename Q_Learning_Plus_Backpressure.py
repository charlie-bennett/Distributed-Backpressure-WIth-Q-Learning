import math
import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
def distance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
def factorial(x):
    if x<=1:
        return 1
    return factorial(x-1)*x


# EDIT THESE PARAMETERS

Random_Walk = 1
RNG = 3
Mu = 1000
Lambda = .2666


def inside_range(ranges1, ranges2):
    count = 0
    for range1, range2 in zip(ranges1, ranges2):
        for outer in [range1, range2]:
            for inner in [range1, range2]:
                for test in inner:
                    if ((test < outer[1]) and (test > outer[0])):
                        count += 1
    if (count >= 4):
        return True
    return False


#must check source is not same as dest!!!

def paths_overlap(start1, finish1, start2, finish2):
    slopes = [0.0 for dummy in range(2)]
    y_intercepts = [0.0 for dummy in range(2)]
    y_ranges = [[0, 0] for dummy in range(2)]
    x_ranges = [[0, 0] for dummy in range(2)]
    starts = [start1, start2]
    finishes = [finish1, finish2]
    for i in range(2):
        y_ranges[i] = [min(starts[i][1], finishes[i][1]), max(starts[i][1], finishes[i][1])]
        x_ranges[i] = [min(starts[i][0], finishes[i][0]), max(starts[i][0], finishes[i][0])]
        rise = float(finishes[i][1] - starts[i][1])
        run = float(finishes[i][0] - starts[i][0])
        slopes[i] = float(rise / run) if run != 0 else float('inf')
        y_intercepts[i] = float(starts[i][1] - (starts[i][0] * slopes[i]))


    for i in [0, 1]:
        for j in [0, 1]:
            if j == i:
                continue

            if slopes[i] == slopes[j]:
                if y_intercepts[i] == y_intercepts[j]:
                    # do y ranges overlap
                    _y_test = [(_test>y_ranges[i][0]) and (_test<y_ranges[i][1]) for _test in y_ranges[j]]
                    if sum(_y_test) > 0:
                        _x_test = [(_test >= x_ranges[i][0]) and (_test<=x_ranges[i][1]) for _test in x_ranges[j]]
                        if sum(_x_test) > 0:
                            return True

                    _y_test = [(_test >= y_ranges[i][0]) and (_test <= y_ranges[i][1]) for _test in y_ranges[j]]
                    if sum(_y_test) > 0:
                        _x_test = [(_test>x_ranges[i][0]) and (_test<x_ranges[i][1]) for _test in x_ranges[j]]
                        if sum(_x_test) > 0:
                            return True

                return False
            else:
                # I am verticle note, what about right angle?
                if slopes[i] == float('inf'):
                    vert_x_value = starts[i][0]
                    if ((vert_x_value > (x_ranges[j][0]+.01)) and (vert_x_value < (x_ranges[j][1]-.01))):
                        vert_y_value = (vert_x_value * slopes[j]) + y_intercepts[j]
                        if (vert_y_value > (y_ranges[i][0] + .01)) and (vert_y_value < (y_ranges[i][1]-.01)):
                            if (vert_y_value > (y_ranges[j][0] + .01)) and (vert_y_value < (y_ranges[i][1]-.01)):
                                return True
                    return False
                if slopes[i] == 0:
                    hor_y_value = starts[i][1]
                    if ((hor_y_value > (y_ranges[j][0] + .01)) and (hor_y_value < (y_ranges[j][1] - .01))):
                        hor_x_value = (hor_y_value - y_intercepts[j]) / slopes[j]
                        if ((hor_x_value > (x_ranges[i][0]+.01)) and (hor_x_value < x_ranges[i][1])):
                            if ((hor_x_value > (x_ranges[j][0] + .01)) and (hor_x_value < (x_ranges[j][1] - .01))):
                                return True
                    return False
                if (slopes[j] == 0) or (slopes[j] == float('inf')):
                    continue
                x_crash = (float(y_intercepts[j] - y_intercepts[i]) / float(slopes[i] - slopes[j]))

                if ((x_crash < (x_ranges[i][1] - .01 )) and (x_crash > (x_ranges[i][0] +.01))):

                    y_crash = (x_crash*slopes[i]) + y_intercepts[i]

                    if ((x_crash < (x_ranges[j][1] - .01)) and (x_crash > (x_ranges[j][0] + .01))):
                        return True
    return False



start1 = tester.positions[6]
stop1 = tester.positions[14]
start2 = tester.positions[4]
stop2 = tester.positions[14]
paths_overlap(start2, stop2, start1, stop1)

N = 5
nodes = 15
num_states = nodes
discount_factor = .6#.95 #.8 # .5 looked pretty good
learning_rate = .3 # was .3
exploration = 5#10 # out of 100, 10 was working ok
saturateValueHigh = 2
saturateValueLow = 0
max_cont_state = 200
Fairness = 0
Fairness_Threshold = 10000
Distance_Scale = 0
Global_Collision_Penalty = 1
Gloabl_Collision_Penalty_Weight = .1
Discount_Time = 1
Sink_QValues = 10
Sink_Reward_Scale = 3
Power_Distance_Scale = 0
Dest_Scaling = 0
moving_average_length = 100


poissonDist =  np.random.poisson(Lambda, 10000)
minTargetValue = .001 # minimum probobility in distrobution we care about






class NetworkGraph(object):
    def __init__(self):
        self.nodes = 15
        self.potential_colors = ["red", "blue", "green", "purple", "orange", "pink", "black"]
        self.num_states = self.nodes
        self.positions = [[0, 0], [1, 0], [3, 0], [4, 0], [3, 1], [1, 3], [1, 1], [3, 3], [0, 4], [1, 4], [2, 4], [3, 4], [2, 2], [4, 4],[2, 0]]
        self.colors = [self.potential_colors[random.randint(0, len(self.potential_colors)-1)] for dummy in range(self.nodes)]
        self.total_sent = 0
        self.total_arrivals = 0
        self.states = [0 for i in range(len(self.positions))]
        self.cont_state_counts = [0 for i in range(len(self.positions))]
        self.Q_values = [[[[1, 1] if (outer != (len(self.positions)-1)) else [Sink_QValues, Sink_QValues] for inner_inner in range(num_states)] for inner in range(len(self.positions))] for outer in
                    range(len(self.positions))]  # [from, to]


        self.U_values = [0 for dummy in range(len(self.positions))]
        self.choices = [0 for i in range(len(self.positions))]
        self.arrivals = []

        self.collision_map = [[0 for inner in range(N)] for outer in range(N)]
        self.collisions = [0 for inner in range(len(self.positions))]
        self.past_collisions = [0 for inner in range(len(self.positions))]
        self.tooFar = [0 for inner in range(len(self.positions))]

        self.totalSunk = 0
        self.epochs = 0

        self.output_path = os.path.expanduser("~/Desktop/test_outputs")
        self.ax = None
        self.fig = None
        self.max_possible_sink = 0
        self.moving_average_sink = []
        self.U_history = []
        self.moving_average_sink_history = []





    def doIteration(self, epoch):
        self.U_history.append([val for val in self.U_values])
        self.moving_average_sink_history.append(sum(self.moving_average_sink)/moving_average_length)
        # past collisions gets current colisions
        self.past_collisions = [collision for collision in self.collisions]
        # Update Random Walk If Applicable
        for i in range(len(self.positions)):
            if (random.randint(0, Mu) == 1) and Random_Walk:
                delta_x = random.randint(-1, 1)
                delta_y = random.randint(-1, 1)
                self.positions[i][0] = max(min(self.positions[i][0] + delta_x, N - 1), 0)
                self.positions[i][1] = max(min(self.positions[i][1] + delta_y, N - 1), 0)
                print("\n\n\n")
                viable = [1 for i in range(len(self.positions))]
                #output = self.get_src_conflicts(0, self.positions[-1], self.positions, viable)
                #self.max_possible_sink = sum(output)



        for i in range(len(self.positions)):
            col_index = self.past_collisions[i]
            potential_weights = [self.Q_values[i][j][self.states[i]][col_index] * (self.U_values[i] - self.U_values[j]) for j in range(len(self.positions))]
            max_weight = max(potential_weights)

            if max_weight <= 0:
                self.choices[i] = i
            else:
                self.choices[i] = potential_weights.index(max_weight)
            if random.randint(1, 100) <= exploration:
                self.choices[i] = random.randint(0, len(self.choices) - 1)
            if self.cont_state_counts[i] > max_cont_state:
                self.choices[i] = random.randint(0, len(self.choices) - 1)


            if Fairness == 1:
                if self.U_values[i] < ((sum(self.U_values) / len(self.U_values)) - Fairness_Threshold):
                    self.choices[i] = i  # revert to mean



            if self.U_values[i] == 0:  # You cannot send a packet if you do not have anything to send
                self.choices[i] = i

        self.choices[-1] = (len(self.choices) - 1)  # Sink Does not Queue Values


        # Get Poisson Arrivals
        self.arrivals = [int(poissonDist[random.randint(0, len(poissonDist)-1)]) for dummy in range(len(self.positions))]

        #reset values
        self.collision_map = [[0 for inner in range(N)] for outer in range(N)]
        self.collisions = [0 for inner in range(len(self.positions))]
        self.tooFar = [0 for inner in range(len(self.positions))]


        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                if i == j:
                    continue
                if (self.choices[i] == i) or (self.choices[j] == j):
                    continue
                if paths_overlap(self.positions[i], self.positions[self.choices[i]], self.positions[j], self.positions[self.choices[j]]):
                    self.collisions[i] = 1

        # slopes = [0.0 for dummy in range(len(self.positions))]
        # y_intercepts = [0.0 for dummy in range(len(self.positions))]
        # y_ranges = [[0, 0] for dummy in range(len(self.positions))]
        # x_ranges = [[0, 0] for dummy in range(len(self.positions))]
        # for i in range(len(self.positions)):
        #     y_ranges[i] = [min(self.positions[self.choices[i]][1], self.positions[i][1]),
        #                    max(self.positions[self.choices[i]][1], self.positions[i][1])]
        #     x_ranges[i] = [min(self.positions[self.choices[i]][0], self.positions[i][0]),
        #                    max(self.positions[self.choices[i]][0], self.positions[i][0])]
        #     rise = float(self.positions[self.choices[i]][1] - self.positions[i][1])
        #     run = float(self.positions[self.choices[i]][0] - self.positions[i][0])
        #     slopes[i] = float(rise/run) if run != 0 else float('inf')
        #
        #     y_intercepts[i] = float(self.positions[i][1] - (self.positions[i][0]*slopes[i]))
        #
        #

        # for i in range(len(self.positions)):
        #     for j in range(len(self.positions)):
        #         if i == j:
        #             continue
        #         if (self.choices[i] == i) or (self.choices[j] == j):
        #             continue
        #         if slopes[i] == slopes[j]:
        #             if y_intercepts[i] == y_intercepts[j]:
        #                 if inside_range([x_ranges[i], y_ranges[i]], [x_ranges[j], y_ranges[j] ]):
        #                     self.collisions[i] = 1
        #         else:
        #             # I am verticle note, what about right angle?
        #             if slopes[i] == float('inf'):
        #                 vert_x_value = self.positions[i][0]
        #                 if ((vert_x_value > x_ranges[j][0]) and (vert_x_value < x_ranges[j][1])):
        #                     vert_y_value = (vert_x_value*slopes[j]) + y_intercepts[j]
        #                     if (vert_y_value > y_ranges[j][0]) and (vert_y_value < y_ranges[j][1]):
        #                         self.collisions[i] = 1
        #                         self.collisions[j] = 1
        #             if slopes[i] == 0:
        #                 hor_y_value = self.positions[i][1]
        #                 if ((hor_y_value > y_ranges[j][0]) and (hor_y_value < y_ranges[j][1])):
        #                     hor_x_value = (hor_y_value - y_intercepts[j])/slopes[j]
        #                     if ((hor_x_value > x_ranges[j][0]) and (hor_x_value < x_ranges[j][1])):
        #                         self.collisions[i] = 1
        #                         self.collisions[j] = 1
        #
        #
        #
        #
        #             #
        #             # if (slopes[i] == float('inf') or (slopes[i] == 0) or (slopes[j] == 0) or (slopes[j] == float('inf'))):
        #             #     if inside_range([x_ranges[i], y_ranges[i]], [x_ranges[j], y_ranges[j] ]):
        #             #         self.collisions[i] = 1
        #
        #
        #
        #             x_crash = (float(y_intercepts[j] - y_intercepts[i]) / float(slopes[i] - slopes[j]))
        #             if ((x_crash < x_ranges[i][1]) and (x_crash > x_ranges[i][0])):
        #                 self.collisions[i] = 1
        #                 print("Crash between %d and %d at X = %f "%(i, j, x_crash))
        #                 print("Range 1x is %.3f, %.3f"%(x_ranges[i][0], x_ranges[i][1]))










        # first sweep
        #
        # for s in range(len(self.positions)):
        #     if self.choices[s] is not s:
        #         d = self.choices[s]
        #         x = [self.positions[s][0], self.positions[s][1]]
        #         y = [self.positions[d][0], self.positions[d][1]]
        #         while distance(x, y) > 1:
        #             next_steps = [
        #                 [[min(max(x[0] + x_delta, 0), N - 1), min(max(x[1] + y_delta, 0), N - 1)] for x_delta in
        #                  range(-1, 2)] for y_delta in
        #                 range(-1, 2)]
        #             next_steps = [col for sub in next_steps for col in sub]
        #             # next_steps = reduce(lambda z, y: z + y, next_steps)
        #             next_steps_distance = [distance(step, y) for step in next_steps]
        #             x = next_steps[next_steps_distance.index(min(next_steps_distance))]
        #             self.collision_map[x[0]][x[1]] += 1
        # #2nd sweep, get collisions
        # collision_points = [[] for dummy in range(len(self.positions))]
        # for s in range(len(self.positions)):
        #     if self.choices[s] is not s:
        #         d = self.choices[s]
        #         x = [self.positions[s][0], self.positions[s][1]]
        #         y = [self.positions[d][0], self.positions[d][1]]
        #         while distance(x, y) > 1:
        #             next_steps = [
        #                 [[min(max(x[0] + x_delta, 0), N - 1), min(max(x[1] + y_delta, 0), N - 1)] for x_delta in
        #                  range(-1, 2)] for y_delta in
        #                 range(-1, 2)]
        #             next_steps = [col for sub in next_steps for col in sub]
        #             next_steps_distance = [distance(step, y) for step in next_steps]
        #             x = next_steps[next_steps_distance.index(min(next_steps_distance))]
        #             if self.collision_map[x[0]][x[1]] > 1:
        #                 self.collisions[s] = 1
        #TX that are too far are not succesfull but still can cause collisions
        for i in range(len(self.positions)):
            self.tooFar[i] = 1 if (distance(self.positions[i], self.positions[self.choices[i]]) > RNG) else 0

        sunk_values = sum([1 if ( self.choices[i] == (len(self.choices)-1)
                        and self.collisions[i] == 0
                        and self.tooFar[i] == 0
                        and self.U_values[i] != 0
                        and i != (len(self.choices) - 1))
                        else 0
                        for i in range(len(self.positions))])

        self.totalSunk += sunk_values

        #Print Stuff
        printing_names = ["Epoch", "Average Throughput", "Sunk Data", "Ave Arrivals", "Ave Sunk", "Sunk Moving Average"]
        ave_sunk = (self.totalSunk/(epoch+1))

        self.moving_average_sink.append(sunk_values)
        if len(self.moving_average_sink) >= moving_average_length:
            self.moving_average_sink = self.moving_average_sink[1:]

        printing_values = [epoch, (self.total_sent / (epoch + 1)), sunk_values, (self.total_arrivals/(epoch+1)),ave_sunk, sum(self.moving_average_sink)/len(self.moving_average_sink)]
        for name, value in zip(printing_names, printing_values):
            print("%s = %.3f"%(name, value))
        print("U Values")
        print(self.U_values)
        print("choices")
        print(self.choices)
        print("collisions")
        print(self.collisions)



        # Update U Values
        for i in range(len(self.positions)-1):
            valid_traffic = [(1 if (not(self.collisions[i] or self.tooFar[i])) else 0) for j in range(len(self.positions))]
            incoming_packets = sum([1 for j in range(len(self.positions)) if ((self.choices[j] == i) and (valid_traffic[j]))])
            incoming_packets += self.arrivals[i]
            good_send = int(not(self.collisions[i] or self.tooFar[i]))
            self.U_values[i] = max(0, self.U_values[i] - good_send)
            self.U_values[i] += incoming_packets # this happens after we send so we dont send data before we recieve
        self.U_values[-1] = 0

        #Update Q values
        rewards = []
        for i in range(len(self.positions)):
            col_index = self.past_collisions[i]
            distance_scale = ((distance(self.positions[i], self.positions[self.choices[i]]) ** Power_Distance_Scale) / (N ** Power_Distance_Scale))
            if ((self.collisions[i] == 0) and (self.tooFar[i] == 0)):

                reward = (1 - distance_scale) if Distance_Scale else 1
                if (Dest_Scaling):
                    reward = max(distance(self.positions[i], self.positions[-1]) - distance(self.positions[self.choices[i]], self.positions[-1]), 0)
                if self.choices[i] == (len(self.positions)-1):
                    reward = reward*Sink_Reward_Scale
                rewards.append(reward)
                if Discount_Time:
                    best_in_next_step = max([self.Q_values[i][j][self.choices[i]][0] for j in range(len(self.positions)) if (j!=i)])
                else:
                    index_other = self.collisions[self.choices[i]]
                    best_in_next_step = max(
                        [self.Q_values[self.choices[i]][j][self.choices[j]][index_other] for j in range(len(self.positions)) if (j != i)])

                self.Q_values[i][self.choices[i]][self.states[i]][col_index] = min(((self.Q_values[i][self.choices[i]][self.states[i]][col_index] *
                    (1 - learning_rate)) + learning_rate * (reward + (discount_factor * best_in_next_step))), saturateValueHigh)

            else:
                reward = (- distance_scale) if Distance_Scale else -1
                rewards.append(reward)
                if Discount_Time:
                    best_in_next_step = max([self.Q_values[i][j][self.choices[i]][1] for j in range(len(self.positions)) if (j!=i)])
                else:
                    best_in_next_step = 0
                self.Q_values[i][self.choices[i]][self.states[i]][col_index] = max(((self.Q_values[i][self.choices[i]][self.states[i]][col_index] *
                    (1 - learning_rate)) + learning_rate * (reward + (
                            discount_factor * self.Q_values[i][self.choices[i]][self.states[self.choices[i]]][1]))), saturateValueLow)
        print("Rewards")
        print(rewards)

        for i in range(len(self.positions)):
            # Update states, state counts & total sent
            if self.states[i] == self.choices[i]:
                self.cont_state_counts[i] += 1
            else:
                self.cont_state_counts[i] = 0

            if self.choices[i] is not i:
                if (not self.collisions[i]) and (not self.tooFar[i]):
                    self.total_sent += 1
            self.states[i] = self.choices[i]
        self.total_arrivals += sum(self.arrivals)


    def setupPlot(self):

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        xmin, xmax, ymin, ymax = 0, N, 0, N
        ticks_frequency = 1

        # Set identical scales for both axes
        self.ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')
        # Set bottom and left spines as x and y axes of coordinate system
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')

        # Remove top and right spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Create 'x' and 'y' labels placed at the end of the axes
        self.ax.set_xlabel('x', size=14, labelpad=15)
        self.ax.set_ylabel('y', size=14, labelpad=15, rotation=0)
        self.ax.xaxis.set_label_coords(1.03, 0.512)
        self.ax.yaxis.set_label_coords(0.5, 1.02)

        # Create custom tick labels
        x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
        x_ticks_major = x_ticks[x_ticks != 0]
        y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
        y_ticks_major = y_ticks[y_ticks != 0]
        self.ax.set_xticks(x_ticks_major)
        self.ax.set_yticks(y_ticks_major)
        self.ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
        self.ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

        # Draw grid lines
        self.ax.grid(which='major', color='grey', linewidth=1, linestyle='-', alpha=0.2)
        self.ax.grid(which='minor', color='grey', linewidth=1, linestyle='-', alpha=0.2)

        # Draw arrows
        self.ax.plot((1), (0), linestyle="", marker=">", markersize=4, color="k",
                transform=self.ax.get_yaxis_transform(), clip_on=False)
        self.ax.plot((0), (1), linestyle="", marker="^", markersize=4, color="k",
                transform=self.ax.get_xaxis_transform(), clip_on=False)


    def print(self):

        color = self.colors

        # Plot points
        x = [given_coordinate[0] for given_coordinate in self.positions]
        y = [given_coordinate[1] for given_coordinate in self.positions]

        sizes = [max(min(5000, given_weight), 50) for given_weight in self.U_values]
        scatter_artist = self.ax.scatter(x, y, facecolors=['none'] * len(color), edgecolors=color, s=sizes)
        annotation_artists = [None for dummy in range(len(self.positions))]
        for i in range(len(self.positions)):
            annotation_artists[i] = self.ax.annotate(i, (x[i], y[i]))


        line_artists = [None for dummy in range(len(self.positions))]
        for i in range(len(self.positions)):
            if i is not self.choices[i]:
                point1 = [self.positions[i][0], self.positions[i][1]]
                point2 = [self.positions[self.choices[i]][0], self.positions[self.choices[i]][1]]
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                if self.tooFar[i]:
                    #line_artists[i] = self.ax.plot(x_values, y_values, ls='dotted', c=color[i], linewidth = .5)
                    line_artists[i] = plt.plot(x_values, y_values, ls='dotted', c=color[i], linewidth=.5)
                    pass
                elif not(self.collisions[i]):
                    line_artists[i] = plt.plot(x_values, y_values, ls="--", c=color[i], linewidth = 1.5)
                else:
                    #line_artists[i] = self.ax.plot(x_values, y_values, ls='--', c=color[i])
                    line_artists[i] = plt.plot(x_values, y_values, ls='-.', c=color[i])

        instant_thorughput = 0
        for i in range(len(self.positions)):
            if self.choices[i] is not i:
                if not self.collisions[i]:
                    instant_thorughput += 1

        printing_names = ["Throughput", "Ave Throughput", "Collisions", "Out of Range", "Epoch", "Average Sink", "Sink Mov Avg" ]
        printing_values = [instant_thorughput, self.total_sent/(self.epochs+1), sum(self.collisions), sum(self.tooFar), self.epochs,
                           self.totalSunk/self.epochs, sum(self.moving_average_sink)/moving_average_length]

        for i in range(len(self.positions)):
            printing_names.append("Q%d"%i)
            printing_values.append(self.U_values[i])
        for i in range(len(self.positions)):
            printing_names.append("Dest %d"%i)
            printing_values.append(self.choices[i])

        text_artists = [None for dummy in range(len(printing_names))]
        for i in range(len(printing_names)):
            text_artists[i] = plt.text(-.1, 1 - (.025 * i), "%s = %.3f" % (printing_names[i], printing_values[i]), horizontalalignment='left',
                     verticalalignment='center', transform=self.ax.transAxes)
        save_path = os.path.join(self.output_path, "epoch_{}.png".format(self.epochs))
        self.fig.savefig(save_path)

        for artist in plt.gca().lines:
            artist.remove()
        for artist in plt.gca().collections:
            artist.remove()
        for artist in line_artists:
            try:
                artist.remove()
            except:
                pass
        for line in self.ax.lines:
            try:
                line.remove()
            except:
                pass
        for artist in text_artists:
            if artist is not None:
                try:
                    artist.remove()
                except:
                    print(artist)
                    pass
        for artist in annotation_artists:
            if artist is not None:
                try:
                    artist.remove()
                except:
                    print(artist)
                    pass


        try:
            Artist.remove(scatter_artist)
        except:
            pass

        #plt.draw()
        #plt.clf()
        plt.close('all')
        #del ax
        #del fig

    def Run(self, num_epochs):
        self.setupPlot()
        for epoch in range(num_epochs):
            self.doIteration(epoch)
            self.epochs += 1
            #if (not(epoch % 100)) and (epoch > num_epochs*.5) and (epoch < num_epochs*.6):
                #self.setupPlot()
                #self.print()
        self.plotHistory()
                #pass
    def plotHistory(self):

        self.fig = plt.figure()
        sink_line, = plt.plot(self.moving_average_sink_history)
        plt.legend([sink_line], ['Moving Average of Number of Sunk Packets'])
        save_path = os.path.join(self.output_path, "Moving_Ave_Sink.png")
        self.fig.savefig(save_path)
        plt.close('all')

        self.fig = plt.figure()
        U_value_lines = [None for i in range(len(self.positions))]
        U_value_legend = [None for i in range(len(self.positions))]
        ind_U_values = [[history[i] for history in self.U_history] for i in range(len(self.positions))]
        for number, U_values in enumerate(ind_U_values):
            U_value_lines[number], = plt.plot(U_values)
            U_value_legend[number] = "Queue of %d"%(number)
        plt.legend(U_value_lines, U_value_legend)
        save_path = os.path.join(self.output_path, "Queue_Values.png")
        self.fig.savefig(save_path)
        plt.close('all')



    def get_src_conflicts(self, iter, dest, sources, viable):
        if iter == (len(sources)-1):
            return viable
        if not viable[iter]:
            return self.get_src_conflicts(iter+1, dest, sources,viable)
        if distance(sources[iter], dest) > RNG:
            print(distance(sources[iter], dest) )
            viable_new = [v for v in viable]
            viable_new[iter] = 0
            return self.get_src_conflicts(iter+1, dest, sources, viable_new)
        else:
            print(distance(sources[iter], dest) )
            print(sources[iter])
            print(dest)
        viable_old = [v for v in viable]
        viable_new = [v for v in viable]
        for i in range(iter+1, len(sources)):
            if not viable_new[i]:
                continue
            if paths_overlap(sources[iter], dest, sources[i], dest):
                viable_new[i] = 0
        viable_greedy = self.get_src_conflicts(iter+1, dest, sources, viable_new)
        viable_next = self.get_src_conflicts(iter+1, dest, sources, viable_new)
        if (sum(viable_greedy) > sum(viable_next)):
            return viable_greedy
        else:
            return viable_next













tester = NetworkGraph()

viable = [1 if i != (len(tester.positions)-1) else 0 for i in range(len(tester.positions))]
tester.Run(100000)




start1 = tester.positions[13]
stop1 = tester.positions[14]
start2 = tester.positions[6]
stop2 = tester.positions[14]
paths_overlap(start2, stop2, start1, stop1)







image_folder = os.path.expanduser("~/Desktop/test_outputs")
video_name = 'video_new.avi'
video_name = os.path.join(image_folder, video_name)


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))
#fourcc = VideoWriter_fourcc(*'MP4V')
#fourcc = 0x7634706d
#video  = cv2.VideoWriter(video_name, fourcc, 20.0, (640,480))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True
video_name_mp4 = os.path.join(image_folder, "video_out_new_new.mp4")

convert_avi_to_mp4(video_name, video_name_mp4)
video_name_mp4 = os.path.join(image_folder, "hourglass_simulation.mp4")

convert_avi_to_mp4(video_name, video_name_mp4)





