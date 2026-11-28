"""ResourceLedger (§6): mass accounts by lot and basis, idempotent operations, no creation of
mass by a transfer.

An account is ``(owner_id, basis)``: a lot (ore body, placer horizon, brine), a source
reservoir, a mobile / dissolved store, an export sink. Every operation has an ``op_id``; a
replayed operation is ignored (T39). The closure per basis:

``initial + generated_from_explicit_source + imported - exported - lost - remaining = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_KINDS = ("initial", "generated", "imported", "exported", "lost")


class LedgerError(ValueError):
    pass


@dataclass
class ResourceLedger:
    tolerance: float = 1e-9  # relative, roundoff only
    balances: dict = field(default_factory=dict)  # (owner, basis) -> amount
    ops: dict = field(default_factory=dict)  # op_id -> record
    totals: dict = field(default_factory=dict)  # basis -> {kind: amount}

    def _tot(self, basis: str) -> dict:
        t = self.totals.get(basis)
        if t is None:
            t = self.totals[basis] = dict.fromkeys(_KINDS, 0.0)
        return t

    def _add(self, key, amount: float) -> None:
        self.balances[key] = self.balances.get(key, 0.0) + amount

    # -- operations ----------------------------------------------------------------------------
    def _new(self, op_id: str, record: dict) -> bool:
        if op_id in self.ops:
            if self.ops[op_id] != record:
                raise LedgerError(f"operation {op_id!r} replayed with different content")
            return False  # idempotent replay: nothing happens
        self.ops[op_id] = record
        return True

    @staticmethod
    def _amount(amount: float) -> float:
        a = float(amount)
        if not a >= 0.0:  # also catches NaN
            raise LedgerError(f"amounts must be >= 0, got {amount!r}")
        return a

    def open(self, op_id: str, owner: str, basis: str, amount: float, kind: str = "initial"):
        """Mass that exists from the start (``initial``) or is formed from an explicit source
        reservoir (``generated``, the source is debited by :meth:`transfer` instead) or comes
        across the open boundary of a region (``imported``)."""
        a = self._amount(amount)
        if kind not in ("initial", "generated", "imported"):
            raise LedgerError(f"unknown opening kind {kind!r}")
        if self._new(op_id, {"op": "open", "owner": owner, "basis": basis, "a": a, "k": kind}):
            self._add((owner, basis), a)
            self._tot(basis)[kind] += a

    def transfer(self, op_id: str, src: str, dst: str, basis: str, amount: float):
        """Move mass between two accounts; never more than the source holds (T38)."""
        a = self._amount(amount)
        rec = {"op": "transfer", "src": src, "dst": dst, "basis": basis, "a": a}
        if op_id in self.ops:
            self._new(op_id, rec)
            return
        have = self.balances.get((src, basis), 0.0)
        if a > have * (1 + self.tolerance) + 1e-300:
            raise LedgerError(f"{src}/{basis}: transfer {a} exceeds the balance {have}")
        self._new(op_id, rec)
        self.balances[(src, basis)] = max(have - a, 0.0)
        self._add((dst, basis), a)

    def remove(self, op_id: str, owner: str, basis: str, amount: float, kind: str = "lost"):
        """Mass leaving the accounted domain: ``exported`` (across the boundary) or ``lost``
        (dissolved to the sea, abraded below the tracked size)."""
        a = self._amount(amount)
        if kind not in ("exported", "lost"):
            raise LedgerError(f"unknown removal kind {kind!r}")
        rec = {"op": "remove", "owner": owner, "basis": basis, "a": a, "k": kind}
        if op_id in self.ops:
            self._new(op_id, rec)
            return
        have = self.balances.get((owner, basis), 0.0)
        if a > have * (1 + self.tolerance) + 1e-300:
            raise LedgerError(f"{owner}/{basis}: removal {a} exceeds the balance {have}")
        self._new(op_id, rec)
        self.balances[(owner, basis)] = max(have - a, 0.0)
        self._tot(basis)[kind] += a

    # -- reading -------------------------------------------------------------------------------
    def balance(self, owner: str, basis: str) -> float:
        return float(self.balances.get((owner, basis), 0.0))

    def remaining(self, basis: str) -> float:
        return float(sum(v for (_o, b), v in self.balances.items() if b == basis))

    def validate(self) -> dict:
        """Closure per basis (``validate_resource_ledger``): residual and ok flag."""
        out = {}
        for basis, t in self.totals.items():
            rem = self.remaining(basis)
            inflow = t["initial"] + t["generated"] + t["imported"]
            resid = inflow - t["exported"] - t["lost"] - rem
            scale = max(inflow, 1e-300)
            out[basis] = {**t, "remaining": rem, "residual": resid,
                          "ok": abs(resid) <= max(self.tolerance * scale, 1e-12)}  # fmt: skip
        neg = [k for k, v in self.balances.items() if v < 0]
        return {"bases": out, "negative_accounts": neg,
                "ok": not neg and all(v["ok"] for v in out.values())}  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "balances": [[o, b, v] for (o, b), v in sorted(self.balances.items())],
            "totals": {b: dict(t) for b, t in sorted(self.totals.items())},
            "operations": len(self.ops),
        }
