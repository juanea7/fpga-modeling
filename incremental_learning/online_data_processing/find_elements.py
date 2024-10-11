"""
Find elements inside lists (or lists os lists)

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2022
Description : Functions on this file return the position of the closest element
              (either above or below) to a given value inside a list (or a list
              of lists).
"""

from bisect import bisect_left


class KeyWrapper:
    """
    This class is used to pass bisect_lest an specific element of the list
    In Python 3.1X.X the bisect module has this feature built-in
    """
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.it[i][self.key]

    def __len__(self):
        return len(self.it)


def get_closest_element_below_item(my_list, my_number, item):
    """
    Assumes my_list is sorted. Returns closest value to my_number.

    If two numbers are equally close, return the smallest number.
    """

    # Apply the wrapper to the list
    wrapped_list = KeyWrapper(my_list, key=item)

    # Get the position of the element inmediately bigger that my_number
    # (or equal if exist in list)
    pos = bisect_left(wrapped_list, my_number)

    # If the return is 0 means all the elements are bigger than my_number
    # or my number is the first element
    # In both cases return intex 0
    if pos == 0:
        return pos
    # If the return index is len(list) means all the values in the list are
    # below my_number, so we return the last element (bigger one)
    if pos == len(wrapped_list):
        return len(wrapped_list)-1

    # In the rest of the cases the index returned could be either:
    # · The index of the closest number above my_number, case in which we
    #   want to return the previous index (the closest number below my number)
    # · The index of my_number in the list, case in which we want to return
    #   that index

    if wrapped_list[pos] > my_number:
        return pos-1
    else:
        return pos


def get_closest_element_below(my_list, my_number):
    """
    Assumes my_list is sorted. Returns closest value to my_number.

    If two numbers are equally close, return the smallest number.
    """

    # Returns the position of the element inmediately bigger that my_number
    # (or equal if exist in list)
    pos = bisect_left(my_list, my_number)

    # If the return is 0 means all the elements are bigger than my_number or
    # my number is the first element
    # In both cases return intex 0
    if pos == 0:
        return pos
    # If the return index is len(list) means all the values in the list are
    # below my_number, so we return the last element (bigger one)
    if pos == len(my_list):
        return len(my_list)-1

    # In the rest of the cases the index returned could be either:
    # · The index of the closest number above my_number, case in which we want
    #   to return the previous index (the closest number below my number)
    # · The index of my_number in the list, case in which we want to return
    #   that index

    if my_list[pos] > my_number:
        return pos-1
    else:
        return pos


def get_closest_element_above(my_list, my_number):
    """
    Assumes my_list is sorted. Returns closest value to my_number.

    If two numbers are equally close, return the smallest number.
    """

    # Returns the position of the element inmediately bigger that my_number
    # (or equal if exist in list)
    pos = bisect_left(my_list, my_number)

    # If the return is 0 means all the elements are bigger than my_number or
    # my number is the first element
    # In both cases return intex 0
    if pos == 0:
        return pos
    # If the return index is len(list) means all the values in the list are
    # below my_number, so we return the last element (bigger one)
    if pos == len(my_list):
        return len(my_list)-1

    # In the rest of the cases the index returned could be either:
    # · The index of the closest number above my_number, case in which we
    #   want to return that index
    # · The index of my_number in the list, case in which we want to return
    #   that index

    return pos
