"""State-machine classification of T1 quartets (pure functions on interface-length series, no I/O).

A quartet is two complementary cell pairs X and Y (e.g. A-B and C-D). Per frame it is in one of
  O  X shares an interface >= L_on and Y does not   (the "original" configuration of the event)
  N  Y shares an interface >= L_on and X does not   (the "new" configuration)
  V  neither pair has one (four-way vertex, very short junctions); a frame with BOTH pairs present is also V
     (flagged `both`, the specification does not cover it)
  X  (written 'x') the quartet could not be measured (a cell is absent)
Hysteresis: a pair counts as present from the first frame with length >= L_on until the length drops below
L_off (< L_on).

Dwell: O and N runs shorter than `n_dwell` frames are absorbed into the surrounding state (V if the surrounding
states differ; into the only neighbour at the start/end of the record). V is exempt from the dwell requirement.
Stable: an O or N run of at least `n_stable` frames. An event starts when the quartet leaves a stable
configuration (enters V or the other configuration) and ends at the START of the first stable configuration
after that; roles are set per event (O = the stable configuration it started from, N = the other pair).

A flip is a change of the O/N sequence ignoring V (O -> N directly or through V). Classes:
  clean               one flip, ends in stable N
  flicker_resolved    >= 3 flips (odd), ends in stable N
  flicker_aborted     >= 2 flips (even), ends in stable O
  abortive_shrink     0 flips, ends in stable O (the quartet passed through V; detected less reliably at coarse
                      time resolution)
  unresolved          no stable configuration before the end of the record, or a cell was absent in the event
Time spent in V never changes clean / flicker classes (a direct O -> N is the same swap as one through V).

LATEST definition (see t1_states_run.DEFINITION for the numbers), added on top of the above:
  group rule     events end when the four cells leave their group (a cell absent or the four no longer one connected contact
                 graph), when a configuration has persisted `n_end` frames (a later swap back is a NEW event), or at the end of the
                 movie (unresolved, not counted); events without a flip are not counted (segment_events_group).
  side rule      while one pair touches, the two flanking cells must each touch both cells of it (all four side pairs in contact
                 in the same frame, any contact), checked at each flip: at the flip frame and at the last frame of the previous
                 O/N state. The first flip failing drops the event; a later failing flip ends the event before it and the event is
                 classed from the valid flips (1 clean, 2 flicker_aborted, 3 flicker_resolved, ...) (apply_side_rule).
  vertex clause  a TRUE four-cell vertex (state V and all four side pairs in contact >= L_off) after the first flip makes an event
                 that would be clean flicker_resolved (vertex_clause). Plain V is not enough: the cells may be apart or in a chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

O, N, V, ABSENT = "O", "N", "V", "x"


@dataclass(frozen=True)
class StateParams:
    l_on: float                # um
    l_off: float               # um, < l_on
    n_dwell: int               # frames; 0 / 1 = no dwell filtering
    n_stable: int              # frames; a configuration this long is stable (an event can start from it)
    n_end: int = 0             # frames; > 0 switches to the group rule: an event ends when the four cells leave their
                               # group or a configuration has persisted this long (see segment_events_group)


def seconds_to_frames(seconds: float, dt_s: float) -> int:
    """Frames needed to last at least `seconds` (a run of k frames lasts k * dt); 0 seconds -> 0 frames.
    The measured frame intervals deviate slightly from their nominal value (e.g. 59.998 s for 60 s), so a tolerance of
    0.02 frames is allowed (the error grows with the number of frames): 120 s at 59.998 s per frame is 2 frames, not 3, and
    480 s at 59.992 s per frame is 8 frames, not 9."""
    if seconds <= 0:
        return 0
    return max(1, int(math.ceil(seconds / dt_s - 0.02)))


def hysteresis(length: np.ndarray, l_on: float, l_off: float) -> np.ndarray:
    """Boolean 'interface present' per frame: switches on at >= l_on, off at < l_off. NaN keeps the previous value."""
    if l_off > l_on:
        raise ValueError("l_off must not exceed l_on")
    out = np.zeros(len(length), dtype=bool)
    on = False
    for i, x in enumerate(length):
        if not np.isnan(x):
            if not on and x >= l_on:
                on = True
            elif on and x < l_off:
                on = False
        out[i] = on
    return out


def assign_states(len_x: np.ndarray, len_y: np.ndarray, l_on: float, l_off: float):
    """Per frame state string array ('O','N','V','x') and a boolean array 'both pairs present'."""
    px, py = hysteresis(len_x, l_on, l_off), hysteresis(len_y, l_on, l_off)
    st = np.full(len(px), V, dtype="<U1")
    st[px & ~py] = O
    st[py & ~px] = N
    both = px & py
    st[np.isnan(len_x) | np.isnan(len_y)] = ABSENT
    return st, both


def runs(states: np.ndarray) -> list[list]:
    """[[state, start, end_exclusive], ...]"""
    out: list[list] = []
    for i, s in enumerate(states):
        if out and out[-1][0] == s:
            out[-1][2] = i + 1
        else:
            out.append([s, i, i + 1])
    return out


def absorb_short(states: np.ndarray, n_dwell: int) -> np.ndarray:
    """Absorb O/N runs shorter than n_dwell: into the common surrounding state, into V if the surrounding states
    differ, into the only neighbour at either end of the record. V and absent frames are exempt. Shortest run first."""
    st = states.copy()
    if n_dwell <= 1:
        return st
    while True:
        rs = runs(st)
        short = [(r[2] - r[1], k) for k, r in enumerate(rs) if r[0] in (O, N) and r[2] - r[1] < n_dwell]
        if not short:
            return st
        _, k = min(short)
        left = next((rs[j][0] for j in range(k - 1, -1, -1) if rs[j][0] != ABSENT), None) if k > 0 and rs[k - 1][0] != ABSENT else None
        right = next((rs[j][0] for j in range(k + 1, len(rs)) if rs[j][0] != ABSENT), None) if k + 1 < len(rs) and rs[k + 1][0] != ABSENT else None
        if left is None and right is None:
            new = V
        elif left is None:
            new = right
        elif right is None:
            new = left
        else:
            new = left if left == right else V
        st[rs[k][1]:rs[k][2]] = new


@dataclass
class Event:
    start: int                       # first frame after leaving the stable configuration
    end: int | None                  # first frame of the next stable configuration (None = unresolved)
    original: str                    # 'O' of this event (the pair it started from): 'O' or 'N' of the input labelling
    final: str | None                # stable state it ended in, None if unresolved
    flips: int
    v_frames: int
    first_new: int | None            # frame of the first state different from the original (None if never)
    sequence: str                    # run-length string of the states between start and end (+ final stable state)
    censored: bool = False           # a cell was absent during the event
    flip_frames: list[int] = field(default_factory=list)
    both_frames: int = 0
    last_obs: int = -1               # last frame of the event in which the quartet was observed (no absent cell)
    end_reason: str = "stable"       # 'stable' (configuration persisted), 'group' (a cell left the 4-cell group), 'movie end'
    pre_frames: int = 0              # length of the stable run the event started from
    post_frames: int = 0             # length of the stable run it ended in (0 if unresolved)
    flips_raw: int = -1              # flips before the side rule (-1 = side rule not applied)
    side_dropped: bool = False       # first flip without side contacts: the event is dropped
    side_truncated: bool = False     # event ended at a flip whose side contacts were missing
    vertex_flag: bool = False        # a true four-cell vertex frame after the first flip (return-to-vertex clause)

    @property
    def klass(self) -> str:
        if self.final is None or self.censored:
            return "unresolved"
        if self.final == self.original:
            return "abortive_shrink" if self.flips == 0 else "flicker_aborted"
        return "clean" if (self.flips == 1 and not self.vertex_flag) else "flicker_resolved"


def segment_events(states: np.ndarray, n_stable: int, both: np.ndarray | None = None) -> list[Event]:
    """Cut the (dwell-filtered) state record of one quartet into events, see the module doc."""
    n_stable = max(1, n_stable)
    rs = runs(states)
    events: list[Event] = []
    stable: str | None = None
    stable_len = 0
    ev_start: int | None = None
    ev_absent = False
    final_len = 0

    def close(end: int | None, final: str | None):
        nonlocal ev_start, ev_absent
        seg = states[ev_start:end] if end is not None else states[ev_start:]
        orig = stable
        # flips: changes of the O/N sequence ignoring V and absent frames, starting from the original configuration
        seq = [orig]
        flip_frames = []
        for i, s in enumerate(seg):
            if s in (O, N) and s != seq[-1]:
                seq.append(s)
                flip_frames.append(ev_start + i)
        if final is not None and final != seq[-1]:
            seq.append(final)
            flip_frames.append(end)
        flips = len(seq) - 1
        flip = {O: N, N: O} if orig == N else {}  # write the sequence in the event's own roles (O = original pair)
        rle = " ".join(f"{flip.get(r[0], r[0])}{r[2] - r[1]}" for r in runs(seg)) if len(seg) else ""
        if final is not None:
            rle += f" |{flip.get(final, final)}"
        seen = np.flatnonzero(seg != ABSENT)
        last_obs = int(ev_start + seen[-1]) if len(seen) else ev_start
        first_new = next((ev_start + i for i, s in enumerate(seg) if s in (O, N) and s != orig), None)
        if first_new is None and final is not None and final != orig:
            first_new = end
        events.append(Event(start=ev_start, end=end, original=orig, final=final, flips=flips,
                            v_frames=int((seg == V).sum()), first_new=first_new, sequence=rle.strip(),
                            censored=ev_absent, flip_frames=flip_frames,
                            both_frames=int(both[ev_start:end].sum()) if both is not None else 0,
                            last_obs=last_obs, pre_frames=stable_len, post_frames=final_len if final is not None else 0))
        ev_start, ev_absent = None, False

    for s, a, b in rs:
        is_stable = s in (O, N) and (b - a) >= n_stable
        if is_stable:
            if ev_start is None and stable is not None and s != stable:
                ev_start = a  # direct O -> N between two frames: the event starts and ends at this frame
            final_len = b - a
            if ev_start is not None:
                close(a, s)
            stable, stable_len = s, b - a
        else:
            if ev_start is not None:
                if s == ABSENT:
                    ev_absent = True
            elif stable is not None and s != ABSENT:
                ev_start = a
    if ev_start is not None:
        close(None, None)
    return events


def segment_events_group(states: np.ndarray, n_start: int, n_end: int, both: np.ndarray | None = None) -> list[Event]:
    """Event segmentation with the group rule. `states` has ABSENT ('x') wherever a cell is missing OR the four cells are
    no longer one connected group.
      start  the quartet leaves a stable configuration (O or N lasting >= n_start frames) into V or the other configuration;
      end    (a) the four cells leave their group (an 'x' run): the event is over, its class follows from the swaps seen
                 and the last observed configuration; (b) a configuration has persisted >= n_end frames: the event ends at
                 the start of that run, and a later swap back from it is a NEW event; (c) the movie ends first: unresolved.
    Everything else (flips, classes) as in segment_events."""
    n_start, n_end = max(1, n_start), max(n_start, n_end)
    rs = runs(states)
    events: list[Event] = []
    stable, stable_len = None, 0
    ev_start = None

    def close(end, final, reason, post=0):
        nonlocal ev_start
        seg = states[ev_start:end] if end is not None else states[ev_start:]
        orig = stable
        seq, flip_frames = [orig], []
        for i, st in enumerate(seg):
            if st in (O, N) and st != seq[-1]:
                seq.append(st)
                flip_frames.append(ev_start + i)
        if reason == "stable" and final != seq[-1]:
            seq.append(final)
            flip_frames.append(end)
        if reason == "group":
            final = seq[-1]
        flips = len(seq) - 1
        flip = {O: N, N: O} if orig == N else {}
        rle = " ".join(f"{flip.get(r[0], r[0])}{r[2] - r[1]}" for r in runs(seg)) if len(seg) else ""
        if final is not None:
            rle += f" |{flip.get(final, final)}" + ("" if reason == "stable" else f" ({reason})")
        seen = np.flatnonzero(seg != ABSENT)
        last_obs = int(ev_start + seen[-1]) if len(seen) else ev_start
        first_new = next((ev_start + i for i, st in enumerate(seg) if st in (O, N) and st != orig), None)
        if first_new is None and reason == "stable" and final != orig:
            first_new = end
        events.append(Event(start=ev_start, end=end, original=orig, final=final if reason != "movie end" else None, flips=flips,
                            v_frames=int((seg == V).sum()), first_new=first_new, sequence=rle.strip(), censored=False,
                            flip_frames=flip_frames, both_frames=int(both[ev_start:end].sum()) if both is not None else 0,
                            last_obs=last_obs, end_reason=reason, pre_frames=stable_len, post_frames=post))
        ev_start = None

    for s, a, b in rs:
        L = b - a
        if s == ABSENT:
            if ev_start is not None:
                close(a, None, "group")
            stable, stable_len = None, 0
            continue
        if ev_start is None:
            if s in (O, N):
                if stable is None:
                    if L >= n_start:
                        stable, stable_len = s, L
                elif s != stable:
                    ev_start = a                       # direct swap between two stable configurations
                    if L >= n_end:
                        close(a, s, "stable", L)
                        stable, stable_len = s, L
            elif stable is not None:                   # V: the quartet left its stable configuration
                ev_start = a
        else:
            if s in (O, N) and L >= n_end:
                close(a, s, "stable", L)
                stable, stable_len = s, L
    if ev_start is not None:
        close(None, None, "movie end")
    return events


def apply_side_rule(ev: Event, st: np.ndarray, side_ok: np.ndarray) -> Event:
    """Neighbour-relationship rule on the flips of an event. `side_ok[t]` = the four side pairs (each cell of one complementary pair
    with each cell of the other) are all in contact in frame t, i.e. the two cells flanking the touching pair both touch both cells of it.
    A flip is valid if side_ok holds at the flip frame and at the last frame of the previous O/N state. With k valid flips from the
    start: k = 0 -> the event is marked side_dropped; k < flips -> the event ends at the failing flip and is classed from the k flips
    (1 clean, 2 flicker_aborted, 3 flicker_resolved, ...); all valid -> unchanged."""
    side_ok = np.asarray(side_ok, bool)
    ff = list(ev.flip_frames)
    k = 0
    for f in ff:
        prev = next((t for t in range(f - 1, -1, -1) if st[t] in (O, N)), None)
        if prev is None or not (side_ok[f] and side_ok[prev]):
            break
        k += 1
    if k == len(ff):
        return ev
    ev.flips_raw = ev.flips
    if k == 0:
        ev.side_dropped, ev.flips, ev.flip_frames = True, 0, []
        return ev
    end = ff[k]
    orig = ev.original
    final = orig if k % 2 == 0 else (N if orig == O else O)
    seg = st[ev.start:end]
    flip = {O: N, N: O} if orig == N else {}
    rle = " ".join(f"{flip.get(r[0], r[0])}{r[2] - r[1]}" for r in runs(seg)) if len(seg) else ""
    rle += f" |{flip.get(final, final)} (side cells)"
    seen = np.flatnonzero(seg != ABSENT)
    ev.end, ev.final, ev.flips, ev.flip_frames = end, final, k, ff[:k]
    ev.end_reason, ev.censored, ev.side_truncated, ev.post_frames = "side cells", False, True, 0
    ev.v_frames = int((seg == V).sum())
    ev.last_obs = int(ev.start + seen[-1]) if len(seen) else ev.start
    ev.sequence = rle.strip()
    return ev


def vertex_clause(ev: Event, st: np.ndarray, side_ok_loff: np.ndarray) -> bool:
    """Return-to-vertex clause: after the first flip the quartet is in a TRUE four-cell vertex in some frame (state V and all four side
    pairs in contact, `side_ok_loff`) before the event ends."""
    if not ev.flip_frames:
        return False
    end = ev.end if ev.end is not None else len(st)
    return any(st[t] == V and bool(side_ok_loff[t]) for t in range(ev.flip_frames[0] + 1, end))


def classify_quartet(len_x, len_y, p: StateParams, group_ok=None, side_ok=None, side_ok_loff=None):
    """Whole pipeline for one quartet: states -> dwell -> events. Returns (events, states, both).
    With p.n_end > 0 and `group_ok` (bool per frame: the four cells are present and form one connected group) the group
    rule of segment_events_group is used, else the stable-time rule of segment_events.
    `side_ok` (bool per frame, see apply_side_rule) switches the neighbour-relationship rule on; `side_ok_loff` the return-to-vertex
    clause (a clean event with a true four-cell vertex after its first flip becomes flicker_resolved). Both default to off."""
    len_x, len_y = np.asarray(len_x, float), np.asarray(len_y, float)
    st, both = assign_states(len_x, len_y, p.l_on, p.l_off)
    st = absorb_short(st, p.n_dwell)
    if p.n_end > 0 and group_ok is not None:
        st = st.copy()
        st[~np.asarray(group_ok, bool)] = ABSENT
        events = segment_events_group(st, p.n_stable, p.n_end, both)
    else:
        events = segment_events(st, p.n_stable, both)
    for ev in events:
        if side_ok is not None and ev.flip_frames:
            apply_side_rule(ev, st, side_ok)
        if side_ok_loff is not None and not ev.side_dropped:
            ev.vertex_flag = vertex_clause(ev, st, side_ok_loff)
    return events, st, both
