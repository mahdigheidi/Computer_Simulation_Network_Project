# Computer Simulation Project - Spring 1402
# Dr. Bardia Safaei

# Mohammad Mahdi Gheidi 98105976
# Zahra Rahmani 99170434

import random
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

class ServicePolicyTypes(Enum):
    FIFO = 'FirstInFirstOut'
    WRR = 'WeightedRoundRobin'
    NPPS = 'NonPreemptivePriorityScheduling'


QUEUE_LIMIT = 10
WRR_QUEUE_LIMIT = [2, 3, 5]
WRR_WEIGHTS = [2, 3, 5]
PROCESSORS_NUM = 3
SERVICE_POLICY = ServicePolicyTypes.WRR
X = 10     # lambda of the host poisson dist
Y = 5      # lambda of the exp distribution of process times
T = 1000   # time of the simulation


class Packet:
    def __init__(self, arrival_time, priority, process_time):
        self.arrival_time = arrival_time
        self.priority = priority
        self.process_time = process_time * 10
        self.queue_time = 0
    def __str__(self):
        return f'Arrival {self.arrival_time}, Priority {self.priority}, Process time {self.process_time}'


class Processor:

    def __init__(self):
        self.busy = False
        self.process_end_time = None
        self.total_run_time = 0


class Router:

    def __init__(self, policy):
        self.queue = []
        self.queue_length = 0
        self.queue_time = 0
        self.queue_count = 0
        self.wrr_queues = [[], [], []] # [High, Mid, Low]
        self.wrr_queue_length = [0, 0, 0]
        self.wrr_queue_time = [0, 0, 0]
        self.wrr_queue_count = [0, 0, 0]
        self.high_priority_queue_times = []
        self.wrr_turn = 0
        self.processors = [Processor() for _ in range(PROCESSORS_NUM)]
        self.policy = policy
        self.dropped_packets = 0

    def process_incoming_packet(self, packet: Packet):

        if self.policy == ServicePolicyTypes.WRR:
            if len(self.wrr_queues[packet.priority]) < WRR_QUEUE_LIMIT[packet.priority]:
                self.wrr_queues[packet.priority].append(packet)
            else:
                self.dropped_packets += 1

        elif self.policy == ServicePolicyTypes.FIFO:
            if len(self.queue) < QUEUE_LIMIT:
                self.queue.append(packet)
            else:
                self.dropped_packets += 1
        elif self.policy == ServicePolicyTypes.NPPS:
            if len(self.queue) < QUEUE_LIMIT:
                self.queue.append(packet)
                self.queue.sort(key=lambda x: x.priority, reverse=True)
            else:
                self.dropped_packets += 1


    def free_unbusy_processors(self, time):
        for processor in self.processors:
            if processor.process_end_time is not None and is_equal(processor.process_end_time, time):
                # print(time)
                processor.busy = False
                processor.process_end_time = None

    def manage_in_queue_packets(self, time):

        free_processors = []

        for processor in self.processors:
            if not processor.busy:
                free_processors.append(processor)

        if len(free_processors) == 0:
            return

        if self.policy == ServicePolicyTypes.WRR:
            if 0 <= self.wrr_turn < WRR_WEIGHTS[0]:
                queue_num = 0 # high
            elif WRR_WEIGHTS[0] <= self.wrr_turn < WRR_WEIGHTS[1]:
                queue_num = 1 # mid
            else:
                queue_num = 2 # low

            if len(self.wrr_queues[queue_num]) > 0:
                packet = self.wrr_queues[queue_num].pop(0)
                packet.queue_time = time - packet.arrival_time

                if packet.priority == 0:
                    self.high_priority_queue_times.append(packet.queue_time)
                self.wrr_queue_time[queue_num] += packet.queue_time
                self.wrr_queue_count[queue_num] += 1
                for processor in self.processors:
                    if not processor.busy:
                        processor.busy = True
                        processor.process_end_time = time + packet.process_time
                        processor.total_run_time += packet.process_time
                        break
            self.wrr_turn += 1
            self.wrr_turn %= sum(WRR_WEIGHTS)

        elif self.policy is ServicePolicyTypes.FIFO or self.policy is ServicePolicyTypes.NPPS:
            for _ in range(min(len(self.queue), len(free_processors))):
                processor = free_processors.pop(0)
                packet = self.queue.pop(0)
                processor.busy = True
                processor.process_end_time = time + packet.process_time
                processor.total_run_time += packet.process_time
                packet.queue_time = time - packet.arrival_time
                self.queue_time += packet.queue_time
                self.queue_count += 1

                if packet.priority == 0:
                    self.high_priority_queue_times.append(packet.queue_time)

    def get_processors_stats(self):
        i = 1
        sum_run_times = 0
        for processor in self.processors:
            print(f'Processor {i} was utilized in {round(processor.total_run_time, 2)}: {round(min(T, processor.total_run_time) * 100 / T, 2)} percent')
            i += 1
            sum_run_times += processor.total_run_time
        
        print('All processors utilization: ', (sum_run_times * 100) / (T * len(self.processors)))

    def update_router_queue_stats(self):
        if self.policy == ServicePolicyTypes.WRR:
            for i in range(3):
                self.wrr_queue_length[i] += len(self.wrr_queues[i])
        else:
            self.queue_length += len(self.queue)

    def get_avg_queue_length(self):
        if self.policy == ServicePolicyTypes.WRR:
            print('Avg length of the high priority queue: ', round(self.wrr_queue_length[0] / (T * 1000), 2))
            print('Avg length of the mid  priority queue: ', round(self.wrr_queue_length[1] / (T * 1000), 2))
            print('Avg length of the low  priority queue: ', round(self.wrr_queue_length[2] / (T * 1000), 2))
            print('Avg length of all router queues: ',
                  round(sum(self.wrr_queue_length) / (T * len(self.wrr_queue_length) * 1000), 2))
        else:
            print('Avg length of the router queue: ', self.queue_length / (T * 1000))

    def get_avg_time_spent_in_queue(self):
        if self.policy == ServicePolicyTypes.WRR:
            try:
                print('Avg time spent in the high priority queue: ',
                        round(self.wrr_queue_time[0] / self.wrr_queue_count[0], 2))
            except ZeroDivisionError:
                print('Avg time spent in the high priority queue: ', 0)

            try:
                print('Avg time spent in the mid  priority queue: ',
                        round(self.wrr_queue_time[1] / self.wrr_queue_count[1], 2))
            except ZeroDivisionError:
                print('Avg time spent in the mid priority queue: ', 0)

            try:
                print('Avg time spent in the low  priority queue: ',
                        round(self.wrr_queue_time[2] / self.wrr_queue_count[2], 2))
            except ZeroDivisionError:
                print('Avg time spent in the low priority queue: ', 0)
            
            print('Avg time spent in all queues: ',
                  round(sum(self.wrr_queue_time) / sum(self.wrr_queue_count), 2))

        else:
            try:
                print('Avg time spent in the queue: ',
                        round(self.queue_time / self.queue_count, 2))
            except ZeroDivisionError:
                print("Cannot process avg time in queue, \
                      because no packets have been processed yet")



def is_equal(x, y):
    return abs(x-y) <= 0.0001



if __name__ == '__main__':
    packets = list()

    last_arrived_time = 0
    k = 0 # Number of packets

    while True:
        interarrival_time = np.random.exponential(X)
        if last_arrived_time+interarrival_time < T:
            last_arrived_time += interarrival_time
            k += 1
        else:
            break

        rnd = random.random()
        if rnd < 0.2:
            priority = 0
        elif 0.2 <= rnd < 0.5:
            priority = 1
        elif rnd >= 0.5:
            priority = 2

        packet_process_time = np.random.exponential(Y)
        packets.append(Packet(round(last_arrived_time, 3), priority, round(packet_process_time, 3)))
        # print(last_arrived_time, priority, packet_process_time)

    # for packet in packets:
    #     print(packet)

    router = Router(SERVICE_POLICY)

    time = 0.0
    while time <= T:
        time += 0.001
        if packets:
            packet = packets[0]

            if is_equal(time, packet.arrival_time):
                router.process_incoming_packet(packets.pop(0))

        router.free_unbusy_processors(time)
        router.manage_in_queue_packets(time)
        router.update_router_queue_stats()


    print('Serving policy of the router: ', router.policy)
    print('Total simulation time: ', T)
    print('Total number of packets sent in the network: ', k)
    print('Number of processed packets: ', k - router.dropped_packets)
    print('Number of dropped packet: ', router.dropped_packets)
    print('Percent of dropped packets', router.dropped_packets * 100 / k)
    router.get_processors_stats()
    router.get_avg_queue_length()
    router.get_avg_time_spent_in_queue()

    cdf_high_priority_queue_times = router.high_priority_queue_times
    for i in range(1, len(cdf_high_priority_queue_times)):
        cdf_high_priority_queue_times[i] += cdf_high_priority_queue_times[i - 1]
    # print(cdf_high_priority_queue_times)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(cdf_high_priority_queue_times, [x+1 for x in range(len(cdf_high_priority_queue_times))])
    fig.savefig('./cdf_high_priority.png')
    plt.close(fig)
    # plt.plot(cdf_high_priority_queue_times)
