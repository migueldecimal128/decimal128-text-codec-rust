# decimal128-text-codec-rust

Thin codec between the two IEEE 754-2019 decimal128 interchange encodings — BID
and DPD — and decimal character sequences, for Rust.

Ported from `decimal128-text-codec-c` for structure and
`decimal128-text-codec-swift` for the boundary — Swift answered first what the
file looks like in a language with a native 128-bit unsigned integer. Both
descend from `decimal128-text-codec-kotlin`, the reference implementation for the
family. One file, no dependencies, `no_std`.

## API

```rust
use decimal128_text_codec::{bid128_to_string, dpd128_to_string, bid128_parse, dpd128_parse};

// bits -> text
let s = bid128_to_string(bits);
let d = dpd128_to_string(bits);

// text -> bits
let bits = bid128_parse("1.23E+45")?;
let peer = dpd128_parse("1.23E+45")?;
```

An encoding is a `u128`: the sign bit is the top bit, and the rest is a
combination field and a trailing significand field laid out per IEEE 754-2019
§3.5.2. That is the entire boundary; there is no value type, no context, no
status flags.

The sibling ports write the canonical positive quiet NaN into an out-parameter
when a parse fails, so that a caller who ignores the failure still holds a legal
IEEE 754-2019 datum. A `#[must_use]` `Result` enforces that at compile time
instead — the value cannot be read without handling the error — so there is no
ignoring caller left to protect. `QNAN` is exported for anyone who wants the
sibling behaviour explicitly: `bid128_parse(text).unwrap_or(QNAN)`.

## Scope

Rounding, where required, is `roundTiesToEven` and is not configurable. Output is
the GDAS `to-scientific-string` form and is not configurable — the exponent sign
is always written (`6.02E+23`), and cohort information is preserved. No
arithmetic, no field extraction, no value type.

Accepted input matches what the full `decimal128-{lang}` ports accept. That
includes the parenthesised NaN payload: **`NaN(123)` as well as `NaN123`**,
matching the spelling C's `strtod` reads. The form is positional and balanced —
`(` immediately after the word, at least one digit, `)` last — so `NaN()`,
`NaN(1`, `NaN(1)2(3)`, `NaN[1]` and `NaN{1}` are all rejected. Rendering never
emits parentheses. The codecs had this before the ports did; the ports took it on
2026-08-23, closing the family's one grammar divergence.

For the full decimal128 implementation see `decimal128-rust` at
<https://decimal128.com>.

## `no_std`

The crate is `#![no_std]` and `#![forbid(unsafe_code)]`. It needs `alloc` only
because the renderers hand back a `String`: the rendering itself runs in a fixed
64-byte stack buffer, since the longest possible output is 42 bytes. Everything
else is `core`. CI builds it for `thumbv7em-none-eabihf` — a bare-metal Cortex-M
target with no operating system — so the claim is checked rather than asserted.

## Requirements

Rust 1.85 (edition 2024). The toolchain is pinned in `rust-toolchain.toml` for
the same reason `decimal128-rust` pins it: a stable release once broke that
repo's next push, which was a pure file rename.

Where this port differs from its siblings, in each case because Rust already had
the answer:

- `u128` is the public boundary, as in Swift and C#. The internals still work in
  `dw1`/`dw0` `u64` pairs exactly as the reference does — split once on entry,
  joined once on exit.
- Parse returns `Result<u128, ParseError>`; the finalize and parse chains return
  tuples rather than writing through out-parameters, as Swift's and Go's do.
- The widening multiply is `u128` arithmetic. The reference decomposes it into
  four 32×32 partial products because the JVM has no unsigned 64-bit
  multiply-high; the C port carries a three-way `__int128` / `__umulh` /
  synthesize-from-limbs split, and an extra build to exercise the fallback.
  Neither has an analogue here.
- Rust panics on overflow in debug and wraps in release, so every site the
  reference relies on wrapping is spelled `wrapping_add` / `wrapping_mul` — the
  same sites the Swift port spells `&+` / `&*`, each carrying a note saying why
  the wrap is the intended arithmetic. Every other site is left checked on
  purpose, so a translation error surfaces as a panic rather than a wrong answer.
- Parsing walks the borrowed UTF-8 bytes of the `&str`. Swift copies them into an
  `Array` first; nothing is copied here.

Internal-consistency claims — the ones unreachable for any input — are spelled
`verify!` and gated on the `verify` cargo feature, the same contract the full
`decimal128-rust` port uses: with the feature off neither the message nor the
condition is compiled. The default is off, so a consumer never takes a panic
inside a library on data they do not control. `demand` — the precision invariant
that stands in for the removed precision-truncation step — is live in every
build.

## Tests

```
git submodule update --init --recursive
cargo test
cargo test --features verify   # the same corpus with every assertion armed
```

| file | what it covers |
|---|---|
| `tests/corpus_vectors.rs` | the four conformance tables over the shared corpus, 2,601 rows, both directions, both formats; then the full 84,783-row canonical table. Also holds the corpus reader and the hex helpers, its only consumer. |

Seven tests, named after the Java, C# and Go ports' seven. That is the whole
suite, and the shape is deliberate: **every test vector in this repo comes from
the shared corpus.** There is no hand-written table. It is an integration test,
so it links the crate from outside and can reach only the public surface.

One assertion the siblings make is absent, and it is a consequence of the
`Result`: they check that a failed parse left the canonical quiet NaN in its
out-parameter. There is no out-parameter here and no stale value a caller could
read — the guarantee is type-level rather than runtime. The bits column of the
`error` table is still asserted, against the exported `QNAN` constant, which
validates that constant on 173 engine-authored rows.

Corpus coverage, measured with `-C instrument-coverage`: **97.1% of lines**. What
is left is the `impossible` path, the failure arms of the assertions, two carry
lines that need a low word of exactly 2⁶⁴−1, and the roll-over-to-10³⁴ branch in
`finalize_subnormal` — which is unreachable in a codec by the same argument that
removed the precision-truncation step, and is left in place because it is
identical code in all seven ports.

The repo is standalone: it depends on no decimal128 port, at runtime or in tests.
The independent oracle is the shared `text_bid_dpd` corpus, generated by the
decimal128-c engine, which reaches its answers by different algorithms than this
codec does.

The vectors are read from `resources/text_bid_dpd`, where `resources` is the
`decimal128-resources` submodule — the same mount every decimal128 port uses.
Nothing is copied into this repo. The directory comes from `CARGO_MANIFEST_DIR`;
an absolute `DECIMAL128_RESOURCES` overrides it.

## License

MIT.
