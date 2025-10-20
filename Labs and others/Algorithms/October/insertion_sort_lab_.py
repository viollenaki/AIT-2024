
"""
Insertion Sort Lab — Trimmed Edition
====================================
Scope:
  • Basic tasks (4): classic sort, descending, counts, strings.
  • Intermediate tasks (2): binary insertion index + binary insertion sort,
    and a generalized insertion sort with key/reverse.

Run tests (xfail until implemented):
  pytest insertion_sort_lab_trimmed.py -q
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional, Sequence, Tuple

# -----------------------------
# Task 1: Basic Insertion Sort
# -----------------------------
def insertion_sort(arr: List[int], *, reverse: bool = False) -> List[int]:
    """Sort integers using insertion sort (no built-ins like sorted()).

    Examples:
        >>> insertion_sort([5,2,9,1,5,6])
        [1,2,5,5,6,9]
        >>> insertion_sort([3,1,2], reverse=True)
        [3,2,1]
    """
    result = arr.copy()
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and ((result[j] > key) if not reverse else (result[j] < key)):
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result


# -----------------------------------
# Task 2: Count Comparisons and Moves
# -----------------------------------
def insertion_sort_with_counts(arr: List[int]) -> Tuple[List[int], int, int]:
    """Return (sorted_list, comparisons, moves).
    comparisons = number of value-to-value order checks.
    moves       = number of element writes due to shifts/placement.
    """
    result = arr.copy()
    comparisons = 0
    moves = 0
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if result[j] > key:
                result[j + 1] = result[j]
                moves += 1
                j -= 1
            else:
                break
        result[j + 1] = key
        moves += 1
    return result, comparisons, moves


# ----------------------------------------
# Task 3: Sort Strings Lexicographically
# ----------------------------------------
def insertion_sort_strings(words: List[str]) -> List[str]:
    """Sort a list of strings lexicographically via insertion sort."""
    result = words.copy()
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result


# --------------------------------
# Intermediate A: Binary Insertion
# --------------------------------
def binary_search_insert_index(a: Sequence[int], key: int, *, lo: int, hi: int) -> int:
    """Find insertion index for key in a[lo:hi] (assume a[lo:hi] is sorted asc).
    Return an index in [lo, hi].
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_insertion_sort(arr: List[int]) -> List[int]:
    """Insertion sort but use binary search to locate insertion index.
    Fewer comparisons; same O(n^2) moves.
    """
    result = arr.copy()
    for i in range(1, len(result)):
        key = result[i]
        idx = binary_search_insert_index(result, key, lo=0, hi=i)
        # Shift elements to make room for key
        for j in range(i, idx, -1):
            result[j] = result[j - 1]
        result[idx] = key
    return result


# --------------------------------------
# Intermediate B: Custom key / reverse
# --------------------------------------
def insertion_sort_custom(
    arr: List[Any],
    *,
    key: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
) -> List[Any]:
    """Generalized insertion sort supporting key= and reverse= like sorted().

    Examples:
        >>> insertion_sort_custom(["aa","b","cccc"], key=len)
        ['b', 'aa', 'cccc']
    """
    result = arr.copy()
    if key is None:
        keyfunc = lambda x: x
    else:
        keyfunc = key
    for i in range(1, len(result)):
        item = result[i]
        item_key = keyfunc(item)
        j = i - 1
        while j >= 0 and ((keyfunc(result[j]) > item_key) if not reverse else (keyfunc(result[j]) < item_key)):
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = item
    return result


# -----------------
# Pytest (xfail)
# -----------------
import pytest

def test_task1_basic():
    assert insertion_sort([5,2,9,1,5,6]) == [1,2,5,5,6,9]
    assert insertion_sort([3,1,2], reverse=True) == [3,2,1]

def test_task2_counts():
    out, comps, moves = insertion_sort_with_counts([3,2,1])
    assert out == [1,2,3]
    assert comps >= 3 and moves >= 3

def test_task3_strings():
    assert insertion_sort_strings(["b","aa","ab"]) == ["aa","ab","b"]

def test_intermediate_a_binary():
    a = [1,3,5,7]
    assert binary_search_insert_index(a, 4, lo=0, hi=4) == 2
    assert binary_insertion_sort([3,1,2,4]) == [1,2,3,4]

def test_intermediate_b_custom():
    words = ["aa","b","cccc"]
    assert insertion_sort_custom(words, key=len) == ["b","aa","cccc"]
    

