# decimal128-text-codec-rust

A thin codec between the two IEEE 754-2019 decimal128 interchange encodings —
BID and DPD — and decimal character sequences. Nothing else: no arithmetic, no
comparison, no value type, no context, no status flags.

One file, no dependencies, `no_std`. The point is that a caller who holds
interchange bits or a string, and wants the other one, can depend on this
without pulling in a full IEEE 754-2019 engine.

For the complete decimal128 implementation see `decimal128-rust` at
<https://decimal128.com>.

## API

```rust
use decimal128_text_codec::{
    MAX_LEN, QNAN, bid128_to_ascii, dpd128_to_ascii, bid128_parse, dpd128_parse,
};

// bits -> text, into the caller's buffer
let mut buf = [0u8; MAX_LEN];
let s: &str = bid128_to_ascii(&mut buf, bits);
let d: &str = dpd128_to_ascii(&mut buf, bits);

// text -> bits, into the caller's u128; None means success
let mut bits = 0u128;
let mut peer = 0u128;
let err: Option<&'static str> = bid128_parse(&mut bits, "1.23E+45");
let _ = dpd128_parse(&mut peer, "1.23E+45");
```

An encoding is a `u128`: the sign bit is the top bit, and the rest is a
combination field and a trailing significand field laid out per IEEE 754-2019
§3.5.2. That is the entire boundary; there is no value type, no context, no
status flags.

The parsers return `None` on success and the failure message on failure,
writing the encoding through an out-parameter — the same shape every sibling
port has: `const char *` in C, `?[]const u8` in Zig, a nullable string in
Swift, Java, Kotlin and C#. On failure they write the canonical positive quiet
NaN, so a caller who ignores the return value still holds a legal
IEEE 754-2019 datum rather than a stale one. `QNAN` is exported so that pattern
can be named. The message is the engine's `ParseStatus` name for the failure —
`EMPTY_STRING`, `INVALID_FORMAT`, `UNEXPECTED_CHAR`, `DOUBLE_DOT`,
`NO_COEFFICIENT_DIGIT`, `NO_EXPONENT_DIGIT`, `INVALID_UNDERSCORE_LOCATION`,
`FAILURE_PARSING_INFINITY`, `FAILURE_PARSING_NAN` — and the conformance suite
asserts it against the shared corpus, so it is part of the contract.

A `Result<u128, ParseError>` would be the more idiomatic Rust surface, and it
would make the error impossible to ignore. It is deliberately not what this
port does: the whole family agrees on what a parse failure is and on what a
failed parse leaves behind, and this port does not get to disagree.

The renderers write into a caller-owned `[u8; MAX_LEN]` and return a borrowed
`&str` into it, so nothing here allocates, copies or frees. `MAX_LEN` is 42,
the exact bound on a rendered decimal128, and it lives in the parameter's type,
so an undersized buffer is a compile error rather than a runtime check and the
renderers keep their cannot-fail property. The C and Zig ports take the
caller's buffer the same way; Swift, Java, Kotlin and C# also offer owned
strings, which is the currency in those languages.

## Behavior

**Rendering cannot fail.** Every 128-bit pattern maps to a valid decimal
character sequence. A non-canonical BID coefficient (≥ 10³⁴) renders as zero at
the encoded exponent, per IEEE 754-2019 §3.5.2, and a NaN payload ≥ 10³³
renders as no payload. Those patterns never arise from this codec's own encode
direction — they arise from network and file data, which is why the codec
handles them.

**Output is the GDAS `to-scientific-string` form, and only that.** Integer form
when the quantum exponent is zero, plain form when the quantum exponent is
negative and the adjusted exponent is at least `-6`, scientific otherwise. The
exponent sign is always written:
`6.02E+23`, never `6.02E23`. Cohort information is preserved — `1.00` and `1.0`
are distinct outputs from distinct encodings of the same numerical value, and
nothing is ever normalized or stripped.

**Accepted input** matches what the full `decimal128-{lang}` ports accept, with
one deliberate exception below, so a caller can move between this codec and the
full library without the grammar shifting under them: plain and scientific
notation with `E` or `e`, a leading `+` or `-`, `_` digit separators *between
two digits*, and the case-insensitive words `Infinity`, `Inf`, `NaN`, `qNaN`,
`sNaN` with an optional sign and an optional payload of decimal digits.

**The exception: a NaN payload may be parenthesised** — `NaN(123)` as well as
`NaN123`. C's own `strtod` reads `NAN(n-char-sequence)`, so the spelling
reaches a caller from the platform itself. The form is positional and balanced:
`(` immediately after the word, at least one digit, `)` last. So `NaN()`,
`NaN(1`, `NaN(1]`, `NaN(1)2(3)`, `NaN (1)`, `NaN[1]` and `NaN{1}` are all
rejected. The resemblance to `strtod` stops at the parentheses — its sequence
is letters, digits and underscores read as an implementation-defined integer,
so `NAN(abc)` and `NAN(0x1c8)` are legal there and rejected here. Rendering
never emits parentheses; `NaN(123)` parses and renders back as `NaN123`.

The full ports accept the same spelling — `decimal128-rust` took it on
2026-08-23, closing what had been the codecs' one grammar divergence from them.

**Rounding is ties-to-even, hardcoded**, and reachable only from the text →
bits direction: more than 34 significant digits, overflow to Infinity, or
underflow to a subnormal or zero. There is no rounding parameter and no format
selector — the premise is a four-function surface with no configuration. IEEE
754-2019 §5.4.2 places `convertFromDecimalCharacter` under the applicable
rounding-direction attribute; this codec deliberately implements only the
ties-to-even case, and is therefore a conforming **subset** of that operation.
Callers needing another rounding direction want the full library.

## Building

```
cargo build
cargo test
cargo test --features verify   # the same corpus with every assertion armed
```

Rust 1.85 (edition 2024). The toolchain is pinned in `rust-toolchain.toml` for
the same reason `decimal128-rust` pins it: a stable release once broke that
repo's next push, which was a pure file rename.

The crate is `#![no_std]` and `#![forbid(unsafe_code)]`, with no allocator at
all. Everything is `core`. CI builds it for `thumbv7em-none-eabihf` — a
bare-metal Cortex-M target with no operating system — so the claim is checked
rather than asserted.

Internal-consistency claims — the ones unreachable for any input — are spelled
`verify!` and gated on the `verify` cargo feature, the same contract the full
`decimal128-rust` port uses: with the feature off neither the message nor the
condition is compiled. The default is off, so a consumer never takes a panic
inside a library on data they do not control. `demand` — the precision
invariant that stands in for the removed precision-truncation step — is live in
every build.

## Tests

Seven tests in one file.

| file | what it covers |
|---|---|
| `tests/corpus_vectors.rs` | the four conformance tables over the shared corpus, 2,601 rows, both directions, both formats; then the full 84,783-row canonical table. Also holds the corpus reader and the hex helpers, its only consumer. |

That is the whole suite, and the shape is deliberate: **every test vector in
this repo comes from the shared corpus.** Nothing in the suite reaches past the
public surface.

Every assertion the siblings make is made here, including the one on the
`error` table's bits columns: that a failed parse left the canonical quiet NaN
in the out-parameter. The out-parameter is seeded with a value that is neither
`QNAN` nor any expected encoding first, so a parser that returned the message
without writing is caught rather than passing on a coincidence. The exported
`QNAN` constant is checked against the same 173 engine-authored rows.

Corpus coverage, measured with `-C instrument-coverage`: **97.1% of lines**.
What is left is the `impossible` path, the failure arms of the assertions, two
carry lines that need a low word of exactly 2⁶⁴−1, and the roll-over-to-10³⁴
branch in `finalize_subnormal` — which is unreachable in a codec by the same
argument that removed the precision-truncation step, and is left in place
because it is identical code in all eight ports.

The repo is standalone: it depends on no decimal128 port, at runtime or in
tests. The independent oracle is the shared `text_bid_dpd` corpus, generated by
the `decimal128-c` engine through its public API; that engine reaches its
answers by different algorithms, so agreement is a cross-check rather than a
tautology.

The vectors are read from `resources/text_bid_dpd`, where `resources` is the
`decimal128-resources` submodule — the same mount every decimal128 port uses.
Nothing is copied into this repo. The directory comes from
`CARGO_MANIFEST_DIR`; an absolute `DECIMAL128_RESOURCES` overrides it.

Two things worth knowing if you touch the tests:

- **The corpus quotes two of its five files.** `parse` and `error` double-quote
  the text column with `\\` and `\"` escapes; the other three leave it bare. A
  naive comma split silently corrupts four rows and still passes, because a
  malformed input fails to parse either way. The reader honours the quoting,
  the row counts are asserted, and `quoted_column_round_trips` checks those
  four rows by hand.
- **The submodule must be checked out.** With `resources/text_bid_dpd` empty
  the loader fails with a message naming the `git submodule update --init
  --recursive` that fixes it, rather than a suite that quietly tests nothing.
  Clone with `--recurse-submodules`.

## Provenance

Ported from `decimal128-text-codec-c` for structure and
`decimal128-text-codec-swift` for the boundary — Swift answered first what the
file looks like in a language with a native 128-bit unsigned integer. Both
descend from `decimal128-text-codec-kotlin`, the reference implementation for
the family. The design spec is `TextCodecLayer.md` in the
`decimal128-workorder` repository.

Where this port differs from its siblings, in each case because Rust already
had the answer:

- **`u128` is the public boundary**, as in Swift, C# and Zig. The internals
  still work in `dw1`/`dw0` `u64` pairs exactly as the reference does — split
  once on entry, joined once on exit.
- **The finalize and parse chains return tuples** internally rather than
  writing through out-parameters, as Swift's and Go's do. The published parsers
  keep the family's out-parameter shape.
- **The widening multiply is `u128` arithmetic.** The reference decomposes it
  into four 32×32 partial products because the JVM has no unsigned 64-bit
  multiply-high; the C port carries a three-way `__int128` / `__umulh` /
  synthesize-from-limbs split, and an extra build to exercise the fallback.
  Neither has an analogue here.
- **Rust panics on overflow in debug and wraps in release**, so every site the
  reference relies on wrapping is spelled `wrapping_add` / `wrapping_mul` — the
  same sites the Swift port spells `&+` / `&*`, each carrying a note saying why
  the wrap is the intended arithmetic. Every other site is left checked on
  purpose, so a translation error surfaces as a panic rather than a wrong
  answer.
- **Parsing walks the borrowed UTF-8 bytes of the `&str`.** Swift copies them
  into an `Array` first; nothing is copied here.

The finalizer's precision-truncation step is removed, in this port and in the
reference alike. It descends from a finalizer built for an arithmetic engine,
whose multiply and FMA hand it a double-width product; a codec's only caller is
a parser that accumulates at most 19 + 15 = 34 significant digits and turns
everything past them into a residue. Measured on the C port: zero executions
across the shared corpus, 400,000 random numerals, and 76,280 finalize calls
from inputs of up to 5,000 digits. A dead path is an unverified path, so it is
replaced by an unconditional `demand`. It stays live in every full port's
`Finalize`, where multiply and FMA do reach it; a caller here that is not a
parser needs it back.

The codec stands on the interchange formats alone. It holds no intermediate
value type: parsing yields loose scalars — sign, quantum exponent, coefficient,
residue — which are finalized and written *toward* an encoding. It does not
pivot through any engine's in-memory representation, and must not be changed
to.

## License

MIT.
