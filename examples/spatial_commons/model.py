"""Spatial commons dynamics, the k-neighbour staying core, and a backup filter.

Implements the ECE532 spatial harvesting game:

* N x N grid, M agents, five absolute moves (stay/up/down/left/right).
* Integer energies: each round costs `metabolic_cost`; eating a ripe apple on
  the tile an agent ends on gives `apple_energy`, clipped to [0, max_energy].
* Ripeness z(q) in [0, 1] grows additively by r times the number of ripe
  tiles among the 8 neighbours, counted in the post-harvest map (Eq. 7). A
  harvested tile resets to 0 and regrows in the same round.
* Failure: any agent at zero energy.

Ripeness is stored exactly as integers: with r = num/den, a tile holds
z * den in {0, ..., den} and is ripe iff it equals den. All safety checks are
therefore exact integer arithmetic with no float tolerance.

Movement (simultaneous, vacating allowed):
  1. Moves off the grid are executed as stay.
  2. Several movers targeting the same tile: the lowest index wins, the rest
     stay.
  3. A mover whose target is occupied by an agent that does not move (it chose
     stay or was itself blocked) stays.
  4. Entering a tile that its occupant is leaving is allowed, so chains resolve,
     but cycles (swaps, 2x2 rotations) are blocked entirely.
  Rules 3-4 are iterated to a fixed point.

Safety (Section 3 of the write-up):
  * Omega_core (k-neighbour staying core, k = `Config.core_k`): every agent is
    on its home tile, at least k of the (border-clipped) 8 neighbours of its
    home are ripe, and E_i > delta * tau_k(z(home_i)) with
    tau_k(z) = ceil((1 - z) / (k r)). All-stay keeps Omega_core invariant
    under Assumption 1: ||h_i - h_j||_inf >= 2 for i != j, T_k * delta <= eta
    and E_max > (T_k - 1) * delta, where T_k = ceil(1 / (k r)).
  * Omega_B (backup set): the states from which rolling out the backup policy
    (walk home, vertical moves first, then stay) reaches Omega_core within
    `horizon` rounds with no starvation. Omega_B contains Omega_core and is
    controlled invariant under the backup policy.
  * SafetyFilter: overrides the fewest agents such that the successor stays in
    Omega_B. The backup action is always feasible, so the filter can always
    find a safe action.
"""

from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import itertools

import numpy as np

STAY, UP, DOWN, LEFT, RIGHT = range(5)
ACTION_NAMES = ("stay", "up", "down", "left", "right")
NUM_ACTIONS = 5
DISPLACEMENTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

Position = tuple[int, int]

DEFAULT_HOMES = ((2, 2), (2, 7), (7, 2), (7, 7))


def _is_int(value) -> bool:
  return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def as_fraction(value) -> Fraction:
  """Parses r given as Fraction, int, float or a string such as '3/200'."""
  if isinstance(value, Fraction):
    return value
  if isinstance(value, float):
    # str() round-trips the shortest decimal, so 0.015 becomes 3/200.
    return Fraction(str(value))
  return Fraction(value)


@dataclasses.dataclass(frozen=True)
class Config:
  """Game constants. Defaults satisfy Assumption 1 with k=4 and T_4=17.

  `core_k` is the neighbour threshold k in {1, ..., 8} of Omega_core.
  """

  grid_size: int = 10
  homes: tuple[Position, ...] = DEFAULT_HOMES
  metabolic_cost: int = 1
  apple_energy: int = 20
  max_energy: int = 40
  initial_energy: int = 20
  growth_rate: Fraction = Fraction(3, 200)
  core_k: int = 4

  def __post_init__(self):
    object.__setattr__(self, "growth_rate", as_fraction(self.growth_rate))
    object.__setattr__(
        self, "homes", tuple((int(r), int(c)) for r, c in self.homes)
    )
    for name in (
        "grid_size",
        "metabolic_cost",
        "apple_energy",
        "max_energy",
        "initial_energy",
    ):
      value = getattr(self, name)
      if not _is_int(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if not _is_int(self.core_k) or not 1 <= self.core_k <= 8:
      raise ValueError("core_k must be an integer in [1, 8]")
    if not 0 < self.growth_rate <= 1:
      raise ValueError("growth_rate must lie in (0, 1]")
    if self.initial_energy > self.max_energy:
      raise ValueError("initial_energy must not exceed max_energy")
    if not self.homes:
      raise ValueError("At least one agent is required")
    if self.num_agents > self.grid_size**2:
      raise ValueError("More agents than tiles")
    for r, c in self.homes:
      if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
        raise ValueError(f"Home {(r, c)} is off the grid")
    if len(set(self.homes)) != len(self.homes):
      raise ValueError("Homes must be distinct")

  @property
  def num_agents(self) -> int:
    return len(self.homes)

  @property
  def ripe_units(self) -> int:
    """Integer value of a ripe tile (the denominator of r)."""
    return self.growth_rate.denominator

  @property
  def growth_units(self) -> int:
    """Integer ripeness added per ripe neighbour per round (numerator of r)."""
    return self.growth_rate.numerator

  @property
  def core_cycle(self) -> int:
    """T_k = ceil(1 / (k r)): rounds between apples for a staying agent."""
    return -(-self.ripe_units // (self.core_k * self.growth_units))

  def core_condition_violations(self) -> tuple[str, ...]:
    """Violations of Assumption 1 (homes and parameters); empty when none.

    Homes may lie on the border; their 8-neighbourhoods are simply clipped.
    """
    problems = []
    for a, b in itertools.combinations(self.homes, 2):
      if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 2:
        problems.append(f"homes {a} and {b} are 8-neighbours")
    t = self.core_cycle
    if t * self.metabolic_cost > self.apple_energy:
      problems.append(
          f"T_k*delta = {t}*{self.metabolic_cost} exceeds eta ="
          f" {self.apple_energy}; need r >= 1/({self.core_k}*floor(eta/delta))"
      )
    if not self.max_energy > (t - 1) * self.metabolic_cost:
      problems.append(
          f"E_max = {self.max_energy} must exceed (T_k-1)*delta ="
          f" {(t - 1) * self.metabolic_cost}"
      )
    return tuple(problems)

  @property
  def core_certified(self) -> bool:
    return not self.core_condition_violations()


@dataclasses.dataclass(frozen=True, eq=False)
class State:
  """Pre-move state at the start of a round.

  `ripeness` holds integer units in [0, config.ripe_units] and is read-only.
  """

  positions: tuple[Position, ...]
  energies: tuple[int, ...]
  ripeness: np.ndarray
  round_index: int = 0

  def __post_init__(self):
    object.__setattr__(
        self, "positions", tuple((int(r), int(c)) for r, c in self.positions)
    )
    object.__setattr__(self, "energies", tuple(int(e) for e in self.energies))
    ripeness = np.array(self.ripeness, dtype=np.int64, copy=True)
    ripeness.flags.writeable = False
    object.__setattr__(self, "ripeness", ripeness)

  @property
  def failed(self) -> bool:
    return min(self.energies) <= 0

  def ripe(self, config: Config) -> np.ndarray:
    return self.ripeness == config.ripe_units

  def __eq__(self, other):
    if not isinstance(other, State):
      return NotImplemented
    return (
        self.positions == other.positions
        and self.energies == other.energies
        and self.round_index == other.round_index
        and np.array_equal(self.ripeness, other.ripeness)
    )


def validate_state(config: Config, state: State) -> None:
  n = config.grid_size
  if len(state.positions) != config.num_agents:
    raise ValueError("Expected one position per agent")
  if len(state.energies) != config.num_agents:
    raise ValueError("Expected one energy per agent")
  for r, c in state.positions:
    if not (0 <= r < n and 0 <= c < n):
      raise ValueError("Agent position off the grid")
  if len(set(state.positions)) != len(state.positions):
    raise ValueError("Agents cannot share a tile")
  if any(not 0 <= e <= config.max_energy for e in state.energies):
    raise ValueError("Energies must lie in [0, E_max]")
  if state.ripeness.shape != (n, n):
    raise ValueError("ripeness must have shape (grid_size, grid_size)")
  if state.ripeness.min() < 0 or state.ripeness.max() > config.ripe_units:
    raise ValueError("ripeness units must lie in [0, ripe_units]")
  if state.round_index < 0:
    raise ValueError("round_index must be nonnegative")


def cross(position: Position) -> tuple[Position, ...]:
  """A tile and its four orthogonal neighbours."""
  r, c = position
  return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def default_state(config: Config) -> State:
  """Section 4.1: agents on their homes, the crosses ripe, all else zero.

  An interior home's cross gives it 4 ripe 8-neighbours and a ripe home tile
  (tau_k = 0), so this state is in Omega_core whenever core_k <= 4.
  """
  n = config.grid_size
  ripeness = np.zeros((n, n), dtype=np.int64)
  for home in config.homes:
    for r, c in cross(home):
      if 0 <= r < n and 0 <= c < n:
        ripeness[r, c] = config.ripe_units
  return State(
      config.homes, (config.initial_energy,) * config.num_agents, ripeness
  )


def validate_actions(config: Config, actions: Sequence[int]) -> tuple[int, ...]:
  if len(actions) != config.num_agents:
    raise ValueError("Expected one action per agent")
  out = []
  for a in actions:
    if not _is_int(a) or not 0 <= a < NUM_ACTIONS:
      raise ValueError(f"Actions must be integers in [0, {NUM_ACTIONS})")
    out.append(int(a))
  return tuple(out)


def resolve_moves(
    grid_size: int,
    positions: Sequence[Position],
    actions: Sequence[int],
) -> tuple[Position, ...]:
  """Simultaneous movement with vacating allowed; see the module docstring."""
  n = len(positions)
  targets = []
  for (r, c), a in zip(positions, actions):
    dr, dc = DISPLACEMENTS[a]
    tr, tc = r + dr, c + dc
    if not (0 <= tr < grid_size and 0 <= tc < grid_size):
      tr, tc = r, c
    targets.append((tr, tc))
  moving = [targets[i] != positions[i] for i in range(n)]

  # Same-target conflicts: the lowest index keeps its move.
  claimed = set()
  for i in range(n):
    if moving[i]:
      if targets[i] in claimed:
        moving[i] = False
      else:
        claimed.add(targets[i])

  occupant = {p: i for i, p in enumerate(positions)}
  while True:
    # Rule 3, to a fixed point: blocked by agents that do not move.
    changed = True
    while changed:
      changed = False
      for i in range(n):
        if moving[i]:
          j = occupant.get(targets[i])
          if j is not None and not moving[j]:
            moving[i] = False
            changed = True
    # Rule 4: the movers form chains and cycles (targets are distinct). Block
    # every cycle, then re-propagate rule 3.
    in_cycle = set()
    for start in range(n):
      if not moving[start] or start in in_cycle:
        continue
      path, seen, i = [], set(), start
      while i is not None and moving[i] and i not in seen:
        seen.add(i)
        path.append(i)
        i = occupant.get(targets[i])
      if i is not None and i in seen:
        in_cycle.update(path[path.index(i) :])
    if not in_cycle:
      break
    for i in in_cycle:
      moving[i] = False

  return tuple(targets[i] if moving[i] else positions[i] for i in range(n))


def ripe_neighbour_counts(ripe: np.ndarray) -> np.ndarray:
  """Number of ripe tiles among each tile's (clipped) 8 neighbours."""
  n = ripe.shape[0]
  padded = np.pad(ripe.astype(np.int64), 1)
  total = np.zeros((n, n), dtype=np.int64)
  for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
      if dr or dc:
        total += padded[1 + dr : 1 + dr + n, 1 + dc : 1 + dc + n]
  return total


@dataclasses.dataclass(frozen=True, eq=False)
class Outcome:
  next_state: State
  eaten: tuple[int, ...]


def transition_from_positions(
    config: Config, state: State, new_positions: Sequence[Position]
) -> Outcome:
  """Harvest, metabolism and regrowth once movement has been resolved."""
  full = config.ripe_units
  ripe = state.ripeness == full
  eaten = tuple(int(ripe[r, c]) for r, c in new_positions)
  energies = tuple(
      max(
          0,
          min(
              config.max_energy,
              e - config.metabolic_cost + config.apple_energy * h,
          ),
      )
      for e, h in zip(state.energies, eaten)
  )
  z_plus = state.ripeness.copy()
  for (r, c), h in zip(new_positions, eaten):
    if h:
      z_plus[r, c] = 0
  counts = ripe_neighbour_counts(z_plus == full)
  z_next = np.minimum(full, z_plus + config.growth_units * counts)
  return Outcome(
      State(new_positions, energies, z_next, state.round_index + 1), eaten
  )


def step(config: Config, state: State, actions: Sequence[int]) -> Outcome:
  """One full round (Section 2.6 steps 4-6) for already-filtered actions."""
  if state.failed:
    raise ValueError("Cannot transition after starvation; reset the episode")
  actions = validate_actions(config, actions)
  new_positions = resolve_moves(config.grid_size, state.positions, actions)
  return transition_from_positions(config, state, new_positions)


def transition(config: Config, state: State, actions: Sequence[int]) -> State:
  return step(config, state, actions).next_state


# ----------------------------------------------------------------------------
# Safety: the k-neighbour staying core and the backup set.


def burn_rounds(config: Config, ripeness_units: int) -> int:
  """tau_k(z) = ceil((1 - z) / (k r)): rounds before a home tile next ripens."""
  remaining = config.ripe_units - int(ripeness_units)
  return -(-remaining // (config.core_k * config.growth_units))


def home_ripe_counts(config: Config, state: State) -> tuple[int, ...]:
  """Ripe tiles among each home's (border-clipped) 8 neighbours."""
  n = config.grid_size
  ripe = state.ripeness == config.ripe_units
  counts = []
  for r, c in config.homes:
    window = ripe[max(r - 1, 0) : min(r + 2, n), max(c - 1, 0) : min(c + 2, n)]
    counts.append(int(window.sum()) - int(ripe[r, c]))
  return tuple(counts)


def in_core(config: Config, state: State) -> bool:
  """Membership in Omega_core (k-neighbour staying core)."""
  if state.failed or state.positions != config.homes:
    return False
  counts = home_ripe_counts(config, state)
  for (r, c), energy, count in zip(config.homes, state.energies, counts):
    if count < config.core_k:
      return False
    tau = burn_rounds(config, state.ripeness[r, c])
    if not energy > config.metabolic_cost * tau:
      return False
  return True


def backup_actions(config: Config, state: State) -> tuple[int, ...]:
  """pi_B: walk home, vertical moves first, then horizontal; stay at home."""
  actions = []
  for (r, c), (hr, hc) in zip(state.positions, config.homes):
    if r < hr:
      actions.append(DOWN)
    elif r > hr:
      actions.append(UP)
    elif c < hc:
      actions.append(RIGHT)
    elif c > hc:
      actions.append(LEFT)
    else:
      actions.append(STAY)
  return tuple(actions)


DEFAULT_HORIZON = 64


def rounds_to_core_reference(
    config: Config, state: State, horizon: int = DEFAULT_HORIZON
) -> int | None:
  """Plain State-based version of rounds_to_core, kept for testing."""
  s = state
  for t in range(horizon + 1):
    if s.failed:
      return None
    if in_core(config, s):
      return t
    if t == horizon:
      return None
    s = transition(config, s, backup_actions(config, s))
  return None


def rounds_to_core(
    config: Config, state: State, horizon: int = DEFAULT_HORIZON
) -> int | None:
  """Backup rollout length to Omega_core, or None if it fails within horizon.

  Returns the first t <= horizon with sigma_t(state) in Omega_core, provided
  no sigma_t' with t' <= t has failed. This is the same rollout as
  rounds_to_core_reference, on raw arrays. It also stops early, with None,
  once no tile is ripe: nothing can ripen again, so Omega_core (which needs
  k >= 1 ripe tiles) is unreachable.
  """
  n, full, g = config.grid_size, config.ripe_units, config.growth_units
  homes, k = config.homes, config.core_k
  delta, eta, e_max = (
      config.metabolic_cost,
      config.apple_energy,
      config.max_energy,
  )
  step_units = k * g
  z = np.array(state.ripeness, dtype=np.int64)
  pad = np.zeros((n + 2, n + 2), dtype=np.int64)  # Ripe map, zero border.
  ripe = pad[1:-1, 1:-1]
  rows = np.empty((n + 2, n), dtype=np.int64)  # Horizontal 3-sums.
  counts = np.empty((n, n), dtype=np.int64)
  positions = state.positions
  energies = list(state.energies)
  for t in range(horizon + 1):
    if min(energies) <= 0:
      return None
    np.equal(z, full, out=ripe, casting="unsafe")
    at_home = positions == homes
    if at_home:
      for (r, c), energy in zip(homes, energies):
        if int(pad[r : r + 3, c : c + 3].sum()) - int(ripe[r, c]) < k:
          break
        if not energy > delta * (-(-(full - int(z[r, c])) // step_units)):
          break
      else:
        return t
    if t == horizon:
      return None
    if not at_home:  # At home, pi_B is all-stay and nobody moves.
      actions = []
      for (r, c), (hr, hc) in zip(positions, homes):
        if r < hr:
          actions.append(DOWN)
        elif r > hr:
          actions.append(UP)
        elif c < hc:
          actions.append(RIGHT)
        elif c > hc:
          actions.append(LEFT)
        else:
          actions.append(STAY)
      positions = resolve_moves(n, positions, actions)
    for i, (r, c) in enumerate(positions):
      if ripe[r, c]:
        energies[i] = max(0, min(e_max, energies[i] - delta + eta))
        z[r, c] = 0
        ripe[r, c] = 0
      else:
        energies[i] = max(0, min(e_max, energies[i] - delta))
    if not ripe.any():
      # No ripe tile: nothing can ever ripen again. The successor can still
      # be in Omega_core only if k = 0, which Config forbids.
      return None
    # 3x3 box sum of the post-harvest ripe map, minus the centre.
    np.add(pad[:, :-2], pad[:, 1:-1], out=rows)
    rows += pad[:, 2:]
    np.add(rows[:-2], rows[1:-1], out=counts)
    counts += rows[2:]
    counts -= ripe
    counts *= g
    z += counts
    np.minimum(z, full, out=z)
  return None


def in_backup_set(
    config: Config, state: State, horizon: int = DEFAULT_HORIZON
) -> bool:
  """Membership in Omega_B."""
  return rounds_to_core(config, state, horizon) is not None


@dataclasses.dataclass(frozen=True, eq=False)
class FilterResult:
  actions: tuple[int, ...]
  overridden: tuple[bool, ...]
  rounds_to_core: int
  candidates_checked: int

  @property
  def intervention_count(self) -> int:
    return sum(self.overridden)


class SafetyFilter:
  """Overrides the fewest agents such that the successor stays in Omega_B.

  Search order: fewer overridden agents first. Among candidates with the same
  number of overrides, ones that give overridden agents their backup action
  come first, then subsets in lexicographic agent order, then actions in
  ACTION_NAMES order. Candidates that resolve to the same positions produce
  the same successor and are checked only once.
  """

  def __init__(self, config: Config, horizon: int = DEFAULT_HORIZON):
    if config.core_condition_violations():
      raise ValueError(
          "Core not certified: " + "; ".join(config.core_condition_violations())
      )
    if not _is_int(horizon) or horizon < 0:
      raise ValueError("horizon must be a nonnegative integer")
    self.config = config
    self.horizon = horizon
    self._certified = None  # Last successor certified to be in Omega_B.

  def candidates(
      self, proposal: tuple[int, ...], backup: tuple[int, ...]
  ) -> list[tuple[int, ...]]:
    """All joint actions in the filter's search order."""
    n = len(proposal)
    ordered = []
    for k in range(n + 1):
      bucket = []
      for subset in itertools.combinations(range(n), k):
        choices = []
        for i in subset:
          alts = [a for a in range(NUM_ACTIONS) if a != proposal[i]]
          if backup[i] != proposal[i]:
            alts.remove(backup[i])
            alts.insert(0, backup[i])
          choices.append(alts)
        for order, repl in enumerate(itertools.product(*choices)):
          non_backup = sum(a != backup[i] for i, a in zip(subset, repl))
          cand = list(proposal)
          for i, a in zip(subset, repl):
            cand[i] = a
          bucket.append((non_backup, subset, order, tuple(cand)))
      bucket.sort(key=lambda item: item[:3])
      ordered.extend(item[3] for item in bucket)
    return ordered

  def __call__(self, state: State, proposal: Sequence[int]) -> FilterResult:
    config = self.config
    proposal = validate_actions(config, proposal)
    # The successor this filter certified last round needs no re-check.
    if not (
        self._certified is not None and state == self._certified
    ) and not in_backup_set(config, state, self.horizon):
      raise ValueError("Safety certification requires a state in Omega_B")
    backup = backup_actions(config, state)
    verdicts: dict[tuple[Position, ...], int | None] = {}
    successors: dict[tuple[Position, ...], State] = {}
    checked = 0
    for cand in self.candidates(proposal, backup):
      positions = resolve_moves(config.grid_size, state.positions, cand)
      if positions not in verdicts:
        checked += 1
        nxt = transition_from_positions(config, state, positions).next_state
        verdicts[positions] = rounds_to_core(config, nxt, self.horizon)
        successors[positions] = nxt
      if verdicts[positions] is not None:
        self._certified = successors[positions]
        return FilterResult(
            cand,
            tuple(a != p for a, p in zip(cand, proposal)),
            verdicts[positions],
            checked,
        )
    # Unreachable: backup(state) leads to a successor in Omega_B.
    raise RuntimeError("No safe joint action found; Omega_B invariance broken")
