import numpy as np
from stable_baselines.common.priority_queue import PriorityQueue


def test_get_priorities():
    """
    Testing the methods: get_priorities(), min(), sum()
    """
    transitions = ["first", "second", "third"]
    td_errors = [0.1, 0.2, 0.3]
    priority_queue = PriorityQueue()
    priority_queue.add_transition(transitions[0], td_errors[0])
    priority_queue.add_transition(transitions[1], td_errors[1])
    priority_queue.add_transition(transitions[2], td_errors[2])

    set_expected_priorities = {1, 1./2, 1./3}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.min()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])


def test_add_transition():

    transitions = ["first", "second", "third"]
    td_errors = [0.1, 0.2, 0.3]
    priority_queue = PriorityQueue()
    priority_queue.add_transition(transitions[2], td_errors[2])
    set_expected_priorities = {1.}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[2]])

    priority_queue.add_transition(transitions[1], td_errors[1])
    set_expected_priorities = {1, 1./2}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[1]])

    priority_queue.add_transition(transitions[0], td_errors[0])
    set_expected_priorities = {1, 1. / 2, 1./3}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])

    # Checking insertion in different order
    priority_queue = PriorityQueue()
    priority_queue.add_transition(transitions[0], td_errors[0])
    set_expected_priorities = {1.}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])

    priority_queue.add_transition(transitions[1], td_errors[1])
    set_expected_priorities = {1, 1. / 2}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])

    priority_queue.add_transition(transitions[2], td_errors[2])
    set_expected_priorities = {1, 1. / 2, 1. / 3}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])


def test_remove_transition():
    transitions = ["first", "second", "third"]
    td_errors = [0.1, 0.2, 0.3]
    priority_queue = PriorityQueue()
    priority_queue.add_transition(transitions[0], td_errors[0])
    priority_queue.add_transition(transitions[1], td_errors[1])
    priority_queue.add_transition(transitions[2], td_errors[2])

    set_expected_priorities = {1, 1./2, 1./3}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])

    priority_queue.remove_transition(transitions[1])
    set_expected_priorities = {1, 1./2}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[0]])

    priority_queue.remove_transition(transitions[0])
    set_expected_priorities = {1}
    assert np.sum(list(set_expected_priorities)) == priority_queue.sum()
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[2]])


def test_pop_transition():

    transitions = ["first", "second", "third"]
    td_errors = [0.1, 0.2, 0.3]
    priority_queue = PriorityQueue()
    priority_queue.add_transition(transitions[0], td_errors[0])
    priority_queue.add_transition(transitions[1], td_errors[1])
    priority_queue.add_transition(transitions[2], td_errors[2])

    priority_queue.remove_transition(transitions[0])
    set_expected_priorities = {1, 1. / 2}
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[1]])

    priority_queue.remove_transition(transitions[1])
    set_expected_priorities = {1}
    assert np.min(list(set_expected_priorities)) == priority_queue.get_priorities([transitions[2]])


if __name__ == '__main__':
    test_get_priorities()
    test_add_transition()
    test_remove_transition()
    test_pop_transition()
