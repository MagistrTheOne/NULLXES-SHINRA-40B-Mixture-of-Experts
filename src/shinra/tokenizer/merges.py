"""Ranked merge admission and leftmost stable heap encoding."""
from dataclasses import dataclass
import heapq
from types import MappingProxyType
from .vocabulary import Vocabulary


@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    output: int


class MergeTable:
    def __init__(self, vocabulary: Vocabulary, merges: tuple[Merge, ...]):
        self.merges = tuple(merges)
        if len(self.merges) != len(vocabulary.payloads) - 256:
            raise ValueError("merge count does not match learned vocabulary")
        ranks = {}
        for rank, merge in enumerate(self.merges):
            output = vocabulary.byte_start + 256 + rank
            if not isinstance(merge, Merge) or any(type(v) is not int for v in (merge.left, merge.right, merge.output)):
                raise ValueError("merge IDs must be integers")
            if merge.output != output or not vocabulary.byte_start <= merge.left < output or not vocabulary.byte_start <= merge.right < output:
                raise ValueError("merge creation order/reference violation")
            pair = (merge.left, merge.right)
            if pair in ranks or vocabulary.payload(output) != vocabulary.payload(merge.left) + vocabulary.payload(merge.right):
                raise ValueError("duplicate pair or inconsistent merge payload")
            ranks[pair] = (rank, output)
        self.ranks = MappingProxyType(ranks)

    def apply(self, ids: list[int]) -> list[int]:
        if not ids:
            return []
        tokens = list(ids)
        previous = [i-1 for i in range(len(ids))]
        following = [i+1 for i in range(len(ids))]
        following[-1] = -1
        alive = [True] * len(ids)
        versions = [0] * len(ids)
        queue = []

        def offer(left):
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0:
                return
            candidate = self.ranks.get((tokens[left], tokens[right]))
            if candidate is not None:
                rank, output = candidate
                heapq.heappush(queue, (rank, left, right, versions[left], versions[right], output))

        for left in range(len(ids)-1):
            offer(left)
        while queue:
            _, left, right, lv, rv, output = heapq.heappop(queue)
            if not alive[left] or not alive[right] or following[left] != right or versions[left] != lv or versions[right] != rv:
                continue
            tokens[left] = output
            versions[left] += 1
            alive[right] = False
            after = following[right]
            following[left] = after
            if after >= 0:
                previous[after] = left
            offer(previous[left])
            offer(left)
        return [token for token, keep in zip(tokens, alive) if keep]


def replace_pair(sequence: tuple[int, ...], pair: tuple[int, int], output: int) -> tuple[int, ...]:
    """One training merge, left to right without overlapping occurrences."""
    result = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index+1]) == pair:
            result.append(output)
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return tuple(result)
