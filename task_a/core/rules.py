"""Scenario A rule definitions: core 5-rule game, extended benchmarks, heldout pools."""

from typing import Callable, Dict, List, Tuple

Triple = Tuple[int, int, int]

# (name, description, check, difficulty)
RuleSpec = Tuple[str, str, Callable[[Triple], bool], str]


class HiddenRule:
    def __init__(
        self,
        name: str,
        description: str,
        check: Callable[[Triple], bool],
        difficulty: str = "medium",
    ):
        self.name = name
        self.description = description  # ground truth, never shown to agent
        self.check = check
        self.difficulty = difficulty

    def validate(self, triple: Triple) -> bool:
        return self.check(triple)


def rules_dict_from_specs(specs: List[RuleSpec]) -> Dict[str, HiddenRule]:
    return {name: HiddenRule(name, desc, check, diff) for name, desc, check, diff in specs}


def _merge_rule_specs(*groups: List[RuleSpec]) -> Dict[str, HiddenRule]:
    out: Dict[str, HiddenRule] = {}
    for group in groups:
        chunk = rules_dict_from_specs(group)
        overlap = set(out) & set(chunk)
        if overlap:
            raise ValueError(f"Duplicate rule ids in specs: {sorted(overlap)}")
        out.update(chunk)
    return out


# --- Spec tables (single source; see ``data/task_a/*/summary.json`` targets) ---

STANDARD_BENCHMARK_SPECS: List[RuleSpec] = [
    (
        "ascending_order",
        "Three numbers in strictly ascending order (a < b < c)",
        lambda t: t[0] < t[1] < t[2],
        "easy",
    ),
    (
        "product_positive",
        "The product of all three numbers is positive",
        lambda t: t[0] * t[1] * t[2] > 0,
        "medium",
    ),
    (
        "range_lte_5",
        "The difference between the largest and smallest number is at most 5",
        lambda t: max(t) - min(t) <= 5,
        "easy",
    ),
    (
        "sum_divisible_by_3",
        "The sum of the three numbers is divisible by 3",
        lambda t: (t[0] + t[1] + t[2]) % 3 == 0,
        "medium",
    ),
    (
        "sum_greater_than_10",
        "The sum of the three numbers is greater than 10",
        lambda t: t[0] + t[1] + t[2] > 10,
        "medium",
    ),
]

EXTRA_BENCHMARK_SPECS: List[RuleSpec] = [
    (
        "exactly_two_positive",
        "Exactly two numbers in the triple are positive",
        lambda t: sum(1 for x in t if x > 0) == 2,
        "medium",
    ),
    (
        "at_least_two_numbers_gt_3",
        "At least two numbers are greater than 3",
        lambda t: sum(1 for x in t if x > 3) >= 2,
        "medium",
    ),
    (
        "median_positive",
        "The median of the three numbers is positive",
        lambda t: sorted(t)[1] > 0,
        "medium",
    ),
    (
        "sum_abs_lte_12",
        "The sum of the absolute values is at most 12",
        lambda t: sum(abs(x) for x in t) <= 12,
        "medium",
    ),
    (
        "sum_of_squares_gt_50",
        "The sum of squares of the three numbers is greater than 50",
        lambda t: sum(x * x for x in t) > 50,
        "medium",
    ),
]

STANDARD_HELDOUT_SPECS: List[RuleSpec] = [
    (
        "all_different",
        "All three numbers are distinct (no repeats)",
        lambda t: len(set(t)) == 3,
        "easy",
    ),
    (
        "first_less_than_last",
        "The first number is strictly less than the last number (a < c)",
        lambda t: t[0] < t[2],
        "easy",
    ),
    (
        "contains_even",
        "At least one number in the triple is even",
        lambda t: any(x % 2 == 0 for x in t),
        "easy",
    ),
    (
        "sum_is_even",
        "The sum of the three numbers is even",
        lambda t: (t[0] + t[1] + t[2]) % 2 == 0,
        "medium",
    ),
    (
        "geometric_ratio",
        "Each number is double the previous one (b = 2a, c = 2b)",
        lambda t: t[0] != 0 and t[1] == 2 * t[0] and t[2] == 2 * t[1],
        "hard",
    ),
]

HELDOUT_HARD_SPECS: List[RuleSpec] = [
    (
        "mountain_or_valley",
        "The numbers form either a 'mountain' (a < b and b > c) or a 'valley' (a > b and b < c).",
        lambda t: (t[0] < t[1] and t[1] > t[2]) or (t[0] > t[1] and t[1] < t[2]),
        "hard",
    ),
    (
        "one_negative_xor_ascending",
        "EITHER exactly one number is negative OR the numbers are strictly ascending, but NOT both.",
        lambda t: bool(sum(1 for x in t if x < 0) == 1) ^ bool(t[0] < t[1] < t[2]),
        "hard",
    ),
    (
        "distinct_xor_same_parity",
        "EITHER all three numbers are distinct OR they all share the same parity (all even or all odd), but NOT both.",
        lambda t: (len(set(t)) == 3) ^ (t[0] % 2 == t[1] % 2 == t[2] % 2),
        "hard",
    ),
    (
        "endpoints_ascending_xor_middle_between",
        "EITHER the first number is strictly less than the last (a < c) OR the middle number is strictly between them (min(a,c) < b < max(a,c)), but NOT both.",
        lambda t: (t[0] < t[2]) ^ (min(t[0], t[2]) < t[1] < max(t[0], t[2])),
        "hard",
    ),
    (
        "sum_even_xnor_product_negative",
        "The sum of the three numbers is even IF AND ONLY IF the product of the three numbers is strictly negative. (Both True or Both False).",
        lambda t: (sum(t) % 2 == 0) == (t[0] * t[1] * t[2] < 0),
        "hard",
    ),
]

MASTER_RULES: Dict[str, HiddenRule] = _merge_rule_specs(
    STANDARD_BENCHMARK_SPECS,
    EXTRA_BENCHMARK_SPECS,
    STANDARD_HELDOUT_SPECS,
    HELDOUT_HARD_SPECS,
)

_CORE_BENCHMARK_NAMES = [spec[0] for spec in STANDARD_BENCHMARK_SPECS]
_EXTENDED_BENCHMARK_NAMES = [spec[0] for spec in STANDARD_BENCHMARK_SPECS + EXTRA_BENCHMARK_SPECS]

# Five-rule view for ``Environment`` / ``evidence_sequences`` (candidate keys in prompts).
RULES: Dict[str, HiddenRule] = {k: MASTER_RULES[k] for k in _CORE_BENCHMARK_NAMES}

# All 5 benchmark rules have mutually exclusive YES-triples: for each rule,
# there exist triples that satisfy ONLY that rule and no others.
BENCHMARK_RULES = list(_CORE_BENCHMARK_NAMES)

# Ten-rule benchmark set for larger-candidate experiments (host-driven, belief_stats, etc.).
EXTENDED_RULES: Dict[str, HiddenRule] = {k: MASTER_RULES[k] for k in _EXTENDED_BENCHMARK_NAMES}
EXTENDED_BENCHMARK_RULES = list(_EXTENDED_BENCHMARK_NAMES)

HELDOUT_RULES: Dict[str, HiddenRule] = {
    spec[0]: MASTER_RULES[spec[0]] for spec in STANDARD_HELDOUT_SPECS
}
HELDOUT_RULES_HARD: Dict[str, HiddenRule] = {
    spec[0]: MASTER_RULES[spec[0]] for spec in HELDOUT_HARD_SPECS
}


def get_rule(name: str) -> HiddenRule:
    try:
        return MASTER_RULES[name]
    except KeyError:
        available = ", ".join(sorted(MASTER_RULES))
        raise ValueError(f"Unknown rule: {name}. Available: {available}") from None


def resolve_heldout_rules(heldout_set: str = "easy") -> Dict[str, HiddenRule]:
    if heldout_set == "easy":
        return dict(HELDOUT_RULES)
    if heldout_set == "hard":
        return dict(HELDOUT_RULES_HARD)
    raise ValueError("heldout_set must be 'easy' or 'hard'")


def all_rules(heldout_set: str = "easy", include_heldout: bool = True) -> Dict[str, HiddenRule]:
    merged: Dict[str, HiddenRule] = {}
    merged.update(EXTENDED_RULES)
    if include_heldout:
        merged.update(resolve_heldout_rules(heldout_set))
    return merged


def list_rules(include_heldout: bool = False) -> Dict[str, str]:
    """Descriptions for the core five benchmark rules (optional easy heldout)."""
    result = {name: rule.description for name, rule in sorted(RULES.items())}
    if include_heldout:
        result.update({name: rule.description for name, rule in sorted(HELDOUT_RULES.items())})
    return result


def list_rules_extended(include_heldout: bool = False, heldout_set: str = "easy") -> Dict[str, str]:
    """Descriptions for the ten extended benchmark rules (optional heldout pool)."""
    result = {name: EXTENDED_RULES[name].description for name in sorted(EXTENDED_RULES)}
    if include_heldout:
        held = resolve_heldout_rules(heldout_set)
        result.update({name: rule.description for name, rule in sorted(held.items())})
    return result
