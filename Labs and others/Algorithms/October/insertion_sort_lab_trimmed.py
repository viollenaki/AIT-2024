from typing import Any, Callable, List, Optional, Sequence, Tuple


def insertion_sort(arr: List[int], *, reverse: bool = False) -> List[int]:
    """Sort integers using insertion sort (no built-ins like sorted())."""
    result = arr.copy()
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and ((result[j] > key) if not reverse else (result[j] < key)):
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result

def insertion_sort_with_counts(arr: List[int]) -> Tuple[List[int], int, int]:
    """Return (sorted_list, comparisons, moves)."""
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

def binary_search_insert_index(a: Sequence[int], key: int, *, lo: int, hi: int) -> int:
    """Find insertion index for key in a[lo:hi] (assume a[lo:hi] is sorted asc)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo

def binary_insertion_sort(arr: List[int]) -> List[int]:
    """Insertion sort but use binary search to locate insertion index."""
    result = arr.copy()
    for i in range(1, len(result)):
        key = result[i]
        idx = binary_search_insert_index(result, key, lo=0, hi=i)
        # Shift elements to make room for key
        for j in range(i, idx, -1):
            result[j] = result[j - 1]
        result[idx] = key
    return result

def insertion_sort_custom(
    arr: List[Any],
    *,
    key: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
) -> List[Any]:
    """Generalized insertion sort supporting key= and reverse= like sorted()."""
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
