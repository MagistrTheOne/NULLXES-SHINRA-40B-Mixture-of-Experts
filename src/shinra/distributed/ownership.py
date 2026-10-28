"""Logical expert ownership independent of physical hardware assignment."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ExpertOwnership:
    owners: tuple[tuple[int, ...], ...]  # [48,128], group-local owner indices
    members: int
    approval_reference: str

    def __post_init__(self):
        if type(self.members) is not int or self.members <= 0 or not isinstance(self.approval_reference, str) or not self.approval_reference.strip():
            raise ValueError("explicit logical membership and approval required")
        if len(self.owners) != 48 or any(len(row) != 128 for row in self.owners):
            raise ValueError("ownership must cover exactly 48*128 experts")
        if any(type(owner) is not int or not 0 <= owner < self.members for row in self.owners for owner in row):
            raise ValueError("invalid logical expert owner")
        object.__setattr__(self, "owners", tuple(tuple(row) for row in self.owners))

    @classmethod
    def from_manifest(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate ownership manifest key")
                result[key] = value
            return result
        body = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
        if not isinstance(body, dict) or set(body) != {"owners", "members", "approval_reference"}:
            raise ValueError("closed ownership manifest schema required")
        return cls(**body)

    @property
    def identity(self):
        payload = json.dumps({"owners": self.owners, "members": self.members,
                              "approval_reference": self.approval_reference}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def local_experts(self, layer: int, rank: int) -> tuple[int, ...]:
        if not 0 <= layer < 48 or not 0 <= rank < self.members:
            raise ValueError("ownership index out of range")
        return tuple(i for i, owner in enumerate(self.owners[layer]) if owner == rank)


@dataclass(frozen=True)
class ParallelLayout:
    ranks: tuple[int, ...]
    coordinates: tuple[tuple[int, int, int, int], ...]  # DP, EP, CP, TP
    expert_owners: tuple[tuple[int, ...], ...]
    approval_reference: str

    def __post_init__(self):
        from itertools import product
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "coordinates", tuple(tuple(c) for c in self.coordinates))
        object.__setattr__(self, "expert_owners", tuple(tuple(c) for c in self.expert_owners))
        if (not self.ranks or len(set(self.ranks)) != len(self.ranks) or
                any(type(r) is not int or r < 0 for r in self.ranks) or len(self.coordinates) != len(self.ranks) or
                any(len(c) != 4 or any(type(v) is not int or v < 0 for v in c) for c in self.coordinates)):
            raise ValueError("explicit unique rank/coordinate inventory required")
        dims = tuple(max(c[i] for c in self.coordinates)+1 for i in range(4))
        if set(self.coordinates) != set(product(*(range(n) for n in dims))) or len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError("logical mesh must be complete and have no duplicate coordinates")
        if dims[3] not in (1,2,4,8):
            raise ValueError("TP degree must preserve eight KV heads")
        ExpertOwnership(self.expert_owners, dims[1], self.approval_reference)

    @property
    def dimensions(self):
        return tuple(max(c[i] for c in self.coordinates)+1 for i in range(4))

    def to_dict(self):
        return {"ranks": list(self.ranks), "coordinates": [list(c) for c in self.coordinates],
                "expert_owners": [list(c) for c in self.expert_owners], "approval_reference": self.approval_reference}

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_manifest(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError("duplicate mesh key")
                result[key] = value
            return result
        body = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
        if not isinstance(body, dict) or set(body) != {"ranks", "coordinates", "expert_owners", "approval_reference"}:
            raise ValueError("closed parallel layout schema required")
        return cls(**body)

    def members(self, rank, role):
        axes = {"world": (0,1,2,3), "tensor": (3,), "expert": (1,), "context": (2,),
                "shared_parameters": (0,1,2), "expert_parameters": (0,2), "source_replicas": (2,3)}
        if role not in axes: raise ValueError("unknown logical group role")
        coordinate = self.coordinates[self.ranks.index(rank)]
        varying = axes[role]
        if role == "world": return self.ranks
        selected = [(c,r) for r,c in zip(self.ranks,self.coordinates)
                    if all(c[i] == coordinate[i] for i in range(4) if i not in varying)]
        return tuple(r for c,r in sorted(selected))

    def inventory(self, config, rank):
        from shinra.core.initialization import parameter_specs
        _, e, _, t = self.coordinates[self.ranks.index(rank)]
        degree = self.dimensions[3]
        result = {}
        for spec in parameter_specs(config):
            if ".moe.experts." in spec.name:
                parts = spec.name.split(".")
                if self.expert_owners[int(parts[1])][int(parts[4])] != e: continue
            axis = None if degree == 1 or spec.role in ("norm", "router") else (1 if spec.name == "embeddings.weight" else 0)
            shape = list(spec.shape)
            start = stop = None
            if axis is not None:
                if shape[axis] % degree: raise ValueError("tensor cannot be evenly partitioned")
                shape[axis] //= degree
                start, stop = t*shape[axis], (t+1)*shape[axis]
            result[spec.name] = {"logical_shape": list(spec.shape), "local_shape": shape, "axis": axis,
                                 "start": start, "stop": stop, "dtype": spec.dtype}
        return result
