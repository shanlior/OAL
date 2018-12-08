import numpy as np
import heapq
import itertools


class PriorityQueue(object):

    def __init__(self):
        """
        Build a Priority queue data structure:

        https://en.wikipedia.org/wiki/Priority_queue

        using a min-heap as in

        https://docs.python.org/3.5/library/heapq.html
        """

        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of transitions to entries
        self.REMOVED = '<removed-transition>'  # placeholder for a removed transition
        self.counter = itertools.count()  # unique sequence count
        self.ranks = {}

    def add_transition(self, transition, td_error=0):
        """
        Add a new transition or update the td_error of an existing transition
        :param transition: (int) transition i
        :param td_error: (float) td-error associated to i
        """
        if transition in self.entry_finder:
            self.remove_transition(transition)
        count = next(self.counter)
        entry = [td_error, count, transition]
        self.entry_finder[transition] = entry
        heapq.heappush(self.pq, entry)

        # find & store rank from array of decreasing priorities
        self.ranks[transition] = heapq.nlargest(len(self.pq), self.pq).index(entry) + 1

        # update rank of following transitions/transitions
        n_transitions = len(self.pq)
        if self.ranks[transition] < n_transitions:
            list_to_update = heapq.nlargest(len(self.pq), self.pq)
            for k in range(self.ranks[transition], n_transitions):
                if list_to_update[k][2] is not self.REMOVED:
                    self.ranks[list_to_update[k][2]] += 1

    def remove_transition(self, transition):
        """
        Mark an existing transition i as REMOVED.  Raise KeyError if not found.
        :param transition: (int) transition i
        """
        # Update preceding transitions, since the data structure is a min-heap
        n_transitions = len(self.pq)
        if self.ranks[transition] < n_transitions:
            list_to_update = heapq.nlargest(len(self.pq), self.pq)
            for k in range(self.ranks[transition], n_transitions):
                if list_to_update[k][2] is not self.REMOVED:
                    self.ranks[list_to_update[k][2]] -= 1
        # Removing the transition from data structures (ranking dict & entry-finder)
        del self.ranks[transition]
        entry = self.entry_finder.pop(transition)
        entry[-1] = self.REMOVED

    def pop_transition(self):
        """
        Remove and return the lowest priority transition. Raise KeyError if empty.
        :returns: (int) transition i with lowesrt priority (inverse of its rank in buffer replay according to td_error)
        """
        while self.pq:
            priority, count, transition = heapq.heappop(self.pq)
            if transition is not self.REMOVED:
                # Updating the preceding transitions, since the data structure is a min-heap
                n_transitions = len(self.pq)
                if self.ranks[transition] < n_transitions:
                    list_to_update = heapq.nlargest(len(self.pq), self.pq)
                    for k in range(self.ranks[transition], n_transitions):
                        if list_to_update[k][2] is not self.REMOVED:
                            self.ranks[list_to_update[k][2]] -= 1
                # Removing from data structures
                del self.ranks[transition]
                del self.entry_finder[transition]
                return transition
        raise KeyError('pop from an empty priority queue')

    def get_priorities(self, transitions):
        """
        A transition's priority p(i) is computed as inverse of the rank of the transition i,
        when the replay-buffer is sorted according to abs(td-error(i))

        :param transitions: ([int]) List of transitions
        :return: ([float]) List of transitions' priorities
        """
        return [1. / self.ranks.get(t, 10e20) for t in transitions]

    def min(self):
        """
        :return: (float) minimal priority value for all transitions
        """
        return 1. / np.max(list(self.ranks.values()))

    def sum(self):
        """
        :return: (float) sum of all Transitions' priorities
        """
        return np.sum(1. / v for v in list(self.ranks.values()))
