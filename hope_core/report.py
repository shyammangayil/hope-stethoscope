"""StethoscopeReport: turns an encoding-loop audit trail and a set of
LayerStates into the human-facing "diagnosis" -- survival ranking, core/slack
partition, capacity heatmap data, and a comparison against a magnitude-
pruning baseline. This module is pure aggregation/formatting over data
that's already been validated elsewhere (kernels/neuron/costs/merge/layer_state);
its own risk profile is "did I compute the right summary statistic", not
"is the underlying math correct".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from hope_core.encoding_loop import ActionRecord, ProgressiveEncodingLoop
from hope_core.neuron import NeuronBank


@dataclass
class NeuronSurvivalEntry:
    layer_name: str
    original_position: int
    capacity: float
    survived: bool  # False if pruned/consumed by a merge at some point
    removed_at_step: int | None
    removal_action: str | None  # "prune" | "merge" | None


@dataclass
class StethoscopeReport:
    initial_banks: dict[str, NeuronBank]
    history: list[ActionRecord]
    initial_total_params: float
    final_total_params: float

    survival_table: list[NeuronSurvivalEntry] = field(default_factory=list, repr=False)

    @classmethod
    def from_loop(cls, loop: ProgressiveEncodingLoop, initial_banks: dict[str, NeuronBank]) -> "StethoscopeReport":
        report = cls(
            initial_banks=initial_banks,
            history=list(loop.history),
            initial_total_params=loop.initial_total_params,
            final_total_params=loop.total_params,
        )
        report._build_survival_table()
        return report

    def _build_survival_table(self) -> None:
        # Track original_position -> stable_id per layer using the same
        # convention LayerState uses at construction (ids 0..N-1 initially).
        removed: dict[str, dict[int, tuple[int, str]]] = {}  # layer -> {stable_id: (step, action)}
        for record in self.history:
            layer_removed = removed.setdefault(record.layer_name, {})
            for sid in record.involved_ids:
                layer_removed[sid] = (record.step, record.action)

        table = []
        for layer_name, bank in self.initial_banks.items():
            capacities = bank.capacity()
            layer_removed = removed.get(layer_name, {})
            for pos in range(bank.n_neurons):
                removal = layer_removed.get(pos)
                table.append(
                    NeuronSurvivalEntry(
                        layer_name=layer_name,
                        original_position=pos,
                        capacity=float(capacities[pos]),
                        survived=removal is None,
                        removed_at_step=removal[0] if removal else None,
                        removal_action=removal[1] if removal else None,
                    )
                )
        self.survival_table = table

    # -- summaries -------------------------------------------------------

    @property
    def final_density(self) -> float:
        if self.initial_total_params <= 0:
            return 0.0
        return self.final_total_params / self.initial_total_params

    def core_slack_partition(self, top_fraction: float = 0.5) -> tuple[list[NeuronSurvivalEntry], list[NeuronSurvivalEntry]]:
        """Split survivors into 'core' (top `top_fraction` by capacity) and
        'slack' (the rest) -- a simple capacity-ranked partition for reporting,
        distinct from DEFT's percentile-threshold elasticity map (deft.py),
        which is the partition actually used for fine-tuning decisions."""
        survivors = sorted((e for e in self.survival_table if e.survived), key=lambda e: -e.capacity)
        cutoff = max(1, int(len(survivors) * top_fraction))
        return survivors[:cutoff], survivors[cutoff:]

    def survival_ranking(self, layer_name: str | None = None) -> list[NeuronSurvivalEntry]:
        """All entries sorted by capacity descending, optionally filtered to one layer."""
        entries = self.survival_table if layer_name is None else [e for e in self.survival_table if e.layer_name == layer_name]
        return sorted(entries, key=lambda e: -e.capacity)

    def capacity_heatmap_data(self) -> dict[str, list[float]]:
        """{layer_name: [capacity_per_original_position]} -- ready to feed a
        plotting library without importing one here (keeps this module's
        dependency footprint minimal; see report_plots.py for matplotlib)."""
        out: dict[str, list[float]] = {}
        for layer_name, bank in self.initial_banks.items():
            caps = bank.capacity()
            out[layer_name] = [float(c) for c in caps]
        return out

    def action_counts(self) -> dict[str, int]:
        counts = {"prune": 0, "merge": 0}
        for record in self.history:
            counts[record.action] += 1
        return counts

    def compare_to_magnitude_baseline(self) -> dict[str, float]:
        """How much of HOPE's pruning decision would a naive |weight|-norm
        baseline have agreed with? Computed only over neurons HOPE actually
        pruned (not merged, since magnitude pruning has no merge analogue) --
        reports the fraction of HOPE-pruned neurons that were ALSO in the
        bottom-k by raw output-weight magnitude in their layer, where k
        matches HOPE's own prune count for that layer. Low agreement is the
        expected/interesting case: it's the whole motivation for capacity
        over magnitude (paper section 1 and section 5's scale-invariance
        discussion) -- magnitude conflates a neuron's raw weight scale with
        its actual functional importance once BN is accounted for.
        """
        pruned_positions: dict[str, list[int]] = {}
        for record in self.history:
            if record.action == "prune":
                pruned_positions.setdefault(record.layer_name, []).append(record.involved_ids[0])

        agreements, total = 0, 0
        for layer_name, ids in pruned_positions.items():
            bank = self.initial_banks[layer_name]
            k = len(ids)
            magnitudes = bank.w_out.flatten(1).norm(dim=-1) if bank.w_out.dim() > 2 else bank.w_out.norm(dim=-1)
            bottom_k_by_magnitude = set(torch.topk(magnitudes, k, largest=False).indices.tolist())
            # NOTE: `ids` here are original bank positions only when no merges
            # have touched this layer before these prunes; for layers with
            # interleaved merges, ids beyond the original N (new parents)
            # aren't comparable to a magnitude baseline and are skipped.
            comparable_ids = [i for i in ids if i < bank.n_neurons]
            for pos in comparable_ids:
                total += 1
                if pos in bottom_k_by_magnitude:
                    agreements += 1

        return {
            "agreement_fraction": (agreements / total) if total else float("nan"),
            "compared_count": total,
        }

    def to_dict(self) -> dict:
        """JSON-serializable summary, for `hope inspect --report out/report.json`."""
        core, slack = self.core_slack_partition()
        return {
            "final_density": self.final_density,
            "initial_total_params": self.initial_total_params,
            "final_total_params": self.final_total_params,
            "action_counts": self.action_counts(),
            "n_survivors": sum(1 for e in self.survival_table if e.survived),
            "n_removed": sum(1 for e in self.survival_table if not e.survived),
            "core_count": len(core),
            "slack_count": len(slack),
            "magnitude_baseline_comparison": self.compare_to_magnitude_baseline(),
            "top_survivors": [
                {"layer": e.layer_name, "position": e.original_position, "capacity": e.capacity}
                for e in self.survival_ranking()[:10]
            ],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
