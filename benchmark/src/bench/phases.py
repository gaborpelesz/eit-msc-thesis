"""Phase-trace parsing.

The forks' vendored `bench_timer.h` writes one line per phase boundary:

    PHASE <name> BEGIN|END <monotonic_ns> [key=value ...]

Spans nest. A phase that repeats -- an image pass, a PatchMatch iteration --
contributes to the total of its name once per span (R-TIM-01, R-TIM-05). The
raw file is kept next to the totals in the run directory; nothing here rewrites
it.
"""

from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    begin_ns: int
    end_ns: int = None
    attrs: dict = field(default_factory=dict)
    children_ns: int = 0

    @property
    def duration_ns(self):
        if self.end_ns is None:
            return None
        return self.end_ns - self.begin_ns


@dataclass
class Trace:
    spans: list = field(default_factory=list)
    unclosed: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    lines: int = 0

    @property
    def begin_ns(self):
        return min((s.begin_ns for s in self.spans), default=None)

    @property
    def end_ns(self):
        ends = [s.end_ns for s in self.spans if s.end_ns is not None]
        return max(ends, default=None)


def _parse_attrs(tokens):
    attrs = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            attrs[key] = value
    return attrs


def parse(text):
    """Parse a phase trace. Malformed lines are recorded, never raised.

    A truncated trace -- the usual outcome of a timeout or a SIGKILL -- leaves
    open spans; they are reported in `unclosed` and contribute to no total,
    because their end time is unknown.
    """
    trace = Trace()
    stack = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("PHASE "):
            continue
        trace.lines += 1
        tokens = line.split()
        if len(tokens) < 4 or tokens[2] not in ("BEGIN", "END"):
            trace.errors.append(f"line {lineno}: malformed: {line!r}")
            continue
        name, kind = tokens[1], tokens[2]
        try:
            timestamp = int(tokens[3])
        except ValueError:
            trace.errors.append(f"line {lineno}: timestamp is not an integer: {line!r}")
            continue
        attrs = _parse_attrs(tokens[4:])

        if kind == "BEGIN":
            stack.append(Span(name=name, begin_ns=timestamp, attrs=attrs))
            continue

        match = next(
            (i for i in range(len(stack) - 1, -1, -1) if stack[i].name == name), None
        )
        if match is None:
            trace.errors.append(f"line {lineno}: END without BEGIN for `{name}`")
            continue
        for orphan in stack[match + 1 :]:
            trace.errors.append(
                f"line {lineno}: `{orphan.name}` was still open when `{name}` ended"
            )
            trace.unclosed.append(orphan)
        del stack[match + 1 :]
        span = stack.pop()
        span.end_ns = timestamp
        span.attrs.update(attrs)
        if stack:
            stack[-1].children_ns += span.duration_ns
        trace.spans.append(span)

    trace.unclosed.extend(stack)
    for span in stack:
        trace.errors.append(f"`{span.name}` never ended (truncated trace)")
    return trace


def parse_file(path):
    try:
        with open(path, errors="replace") as f:
            return parse(f.read())
    except FileNotFoundError:
        return Trace(errors=[f"{path}: no phase trace was written"])


def totals(trace):
    """Per-phase totals, in run.json's stable list-of-structs shape.

    `total_s` sums every span of that name; `self_s` subtracts the time spent
    in nested spans, so summing `self_s` over a trace gives the run without
    double counting.
    """
    rows = {}
    for span in trace.spans:
        row = rows.setdefault(
            span.name, {"name": span.name, "total_s": 0.0, "self_s": 0.0, "count": 0}
        )
        row["total_s"] += span.duration_ns / 1e9
        row["self_s"] += (span.duration_ns - span.children_ns) / 1e9
        row["count"] += 1
    for name in {s.name for s in trace.unclosed}:
        rows.setdefault(name, {"name": name, "total_s": 0.0, "self_s": 0.0, "count": 0})
        rows[name]["unclosed"] = rows[name].get("unclosed", 0) + sum(
            1 for s in trace.unclosed if s.name == name
        )
    for row in rows.values():
        row.setdefault("unclosed", 0)
    return sorted(rows.values(), key=lambda r: -r["total_s"])


def totals_by_pass(trace):
    """Totals split by the method's own `pass=` attribute (R-TIM-07).

    The mapping to the canonical pass vocabulary lives in methods.yaml, so the
    method's own name is what is stored here.
    """
    rows = {}
    for span in trace.spans:
        pass_name = span.attrs.get("pass")
        if pass_name is None:
            continue
        row = rows.setdefault(
            (span.name, pass_name),
            {"name": span.name, "pass": pass_name, "total_s": 0.0, "count": 0},
        )
        row["total_s"] += span.duration_ns / 1e9
        row["count"] += 1
    return sorted(rows.values(), key=lambda r: -r["total_s"])


def format_tree(trace, max_depth=None):
    """Pretty-print the trace as a tree, for `bench phases`."""
    ordered = sorted(trace.spans, key=lambda s: (s.begin_ns, -(s.end_ns or 0)))
    lines = []
    stack = []
    for span in ordered:
        while stack and span.begin_ns >= stack[-1].end_ns:
            stack.pop()
        depth = len(stack)
        if max_depth is None or depth <= max_depth:
            attrs = " ".join(f"{k}={v}" for k, v in sorted(span.attrs.items()))
            lines.append(
                f"{'  ' * depth}{span.name:<{max(1, 34 - 2 * depth)}} "
                f"{span.duration_ns / 1e9:10.3f}s"
                + (f"  {attrs}" if attrs else "")
            )
        stack.append(span)
    for span in trace.unclosed:
        lines.append(f"{span.name:<34} {'unclosed':>11}")
    return "\n".join(lines)
