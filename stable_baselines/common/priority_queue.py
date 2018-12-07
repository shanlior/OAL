import numpy as np
import heapq
import itertools


class PriorityQueue(object):

    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of transitions to entries
        self.REMOVED = '<removed-transition>'      # placeholder for a removed transition
        self.counter = itertools.count()     # unique sequence count
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
            for k in range(self.ranks[transition], n_transitions):
                self.ranks[self.pq[k][2]] += 1

    def remove_transition(self, transition):
        """
        Mark an existing transition i as REMOVED.  Raise KeyError if not found.
        :param transition: (int) transition i
        """
        # Update preceeding transitions, since it's a min-heap
        n_transitions = len(self.pq)
        if self.ranks[transition] < n_transitions:
            for k in range(self.ranks[transition]):
                self.ranks[self.pq[k][2]] -= 1
        # Removing the transition from data structures (ranking dict & entry-finder)
        del self.ranks[transition]
        entry = self.entry_finder.pop(transition)
        entry[-1] = self.REMOVED

    def pop_transition(self):
        """
        Remove and return the lowest priority transition. Raise KeyError if empty.
        """
        while self.pq:
            priority, count, transition = heapq.heappop(self.pq)
            if transition is not self.REMOVED:
                # Updating the preceeding transitions, since min-heap
                n_transitions = len(self.pq)
                if self.ranks[transition] < n_transitions:
                    for k in range(self.ranks[transition]):
                        self.ranks[self.pq[k][2]] -= 1
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
        return 1. / self.ranks[self.pq[0][2]]

    def sum(self):
        return np.sum(1./v for v in list(self.ranks.values()))


"""
if __name__ == "__main__":
    pq_ = PriorityQueue()
    pq_.add_transition("first", 3)
    pq_.add_transition("second", 111)
    pq_.add_transition("third", 0.1)
    #print(heapq.nlargest(len(pq_.pq), pq_.pq))
    print("pq: ", pq_.pq,", priorities: ", pq_.get_priorities(["third", "second", "first"]), ", sum: ",pq_.sum())
    print(["third", "second", "first"])

    pq_.remove_transition("second")
    print("pq: ", pq_.pq, ", priorities: ", pq_.get_priorities(["third", "second", "first"]), ", sum: ",
          pq_.sum())
    print(["third", "second", "first"])
"""