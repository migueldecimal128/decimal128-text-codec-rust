// SPDX-License-Identifier: MIT
#![no_std]
#![forbid(unsafe_code)]

//! Thin codec between the two IEEE 754-2019 decimal128 interchange encodings —
//! BID (Binary Integer Decimal) and DPD (Densely Packed Decimal) — and decimal
//! character sequences.
//!
//! A decimal128 encoding is 128 bits: a sign bit, a combination field and a
//! trailing significand field, laid out per IEEE 754-2019 §3.5.2. It crosses
//! this API as a `u128`. That is the entire boundary; there is no value type,
//! no context, no status flags.
//!
//! This codec provides no arithmetic, no comparison and no field extraction.
//! For the full IEEE 754-2019 decimal128 implementation — arithmetic,
//! comparison, every rounding-direction attribute, IEEE status flags — see
//! <https://decimal128.com>.
//!
//! ## Rounding
//!
//! Rounding, where the text → bits direction requires it, is the IEEE 754-2019
//! `roundTiesToEven` attribute and is not configurable. IEEE 754-2019 §5.4.2
//! places `convertFromDecimalCharacter` under the applicable
//! rounding-direction attribute; this codec deliberately implements only the
//! ties-to-even case and is therefore a conforming subset of that operation.
//! Callers needing another rounding direction want the full library.
//!
//! Output form is the GDAS `to-scientific-string` form and is not
//! configurable. Cohort information is preserved: `"1.00"` and `"1.0"` are
//! distinct outputs from distinct encodings of the same numerical value.
//!
//! ## Round-trip behavior
//!
//! The renderer's output round-trips exactly: parsing rendered text and
//! rendering it again reproduces that text, and rendering a canonical encoding
//! and parsing the result reproduces the original bits. Three classes of input
//! do not round-trip:
//!
//! - **More than 34 significant digits** are accepted and rounded to 34 with
//!   `roundTiesToEven`. The rounded value is encoded; the input text is not
//!   preserved.
//! - **Magnitudes outside the representable range** overflow to `Infinity` or
//!   underflow to a subnormal or zero, per IEEE 754-2019.
//! - **Non-canonical bit patterns** render as something the encoder would
//!   never have emitted, so re-parsing yields the canonical pattern instead.
//!
//! ## Non-canonical bit patterns
//!
//! Not every 128-bit pattern is canonical. Each encoding has its own
//! non-canonical forms — patterns the encoder would never emit — and what
//! makes a pattern non-canonical is a property of that encoding, documented
//! where that encoding is decoded.
//!
//! Neither renderer can fail on one: every 128-bit pattern renders as the
//! value IEEE 754-2019 assigns it. Decoding is one-way for such a pattern,
//! because the parsers emit only canonical encodings, so re-parsing the
//! rendered text yields bits that differ from the original. Non-canonical
//! patterns never arise from this codec's own parsers; they appear only when
//! encodings come from outside, such as network or file data.
//!
//! ## Thread safety
//!
//! All functions are pure and safe for concurrent use.
//!
//! ## This implementation
//!
//! The crate is `#![no_std]` with no allocator at all, and forbids unsafe
//! code. The renderers write into a caller-owned `[u8; MAX_LEN]` and hand back
//! a borrowed `str`, so nothing here allocates, copies or frees. Everything is
//! `core`; nothing needs an operating system.

/*
 * Everything reachable from outside is the four functions at the bottom of this
 * file; the rest is private to the crate.
 */

/* ---------------------------------------------------------------------------
 * Internal-state assertions
 *
 * Two mechanisms: one is a debugging aid, the other a claim that must hold
 * in a shipping build.
 *
 *   verify!(condition)
 *                       internal-consistency claims -- unreachable for ANY
 *                       input, so a failure is a codec bug rather than bad
 *                       data. Emitted only under the `verify` cargo feature,
 *                       which is OFF by default: a consumer compiles this
 *                       crate into their own program and must not take a panic
 *                       inside a library on data they do not control. With the
 *                       feature off the condition is not compiled at all. CI runs the whole corpus with it on.
 *
 *   demand(condition, message)
 *                       claims that must survive into a shipping build: that
 *                       the caller's buffer is at least MAX_LEN, checked once
 *                       on entry to each decoder, and the coefficient-width
 *                       invariant standing in for the removed
 *                       precision-truncation step, where falling through
 *                       would silently encode an out-of-range coefficient.
 *
 * Messages are `&str` literals, so the contracts stay alloc-free.
 * ------------------------------------------------------------------------- */

/// Unconditional abort marking unreachable code.
#[inline(always)]
#[track_caller]
fn impossible(message: &str) -> ! {
    // Returns the never type !.
    panic!("decimal128-text-codec: impossible state: {message}")
}

#[inline(always)]
#[track_caller]
fn demand(condition: bool, message: &str) {
    if !condition {
        impossible(message);
    }
}

macro_rules! verify {
    ($condition:expr) => {{
        #[cfg(feature = "verify")]
        {
            if !($condition) {
                $crate::impossible("verify failed");
            }
        }
    }};
}

/* ------------------------------------------------------------------------
 * Public constants
 * ---------------------------------------------------------------------- */

/// The exact bound on a rendered decimal128, in ASCII character bytes.
///
/// The widest scientific form `-9.999...E-6143` is 1 sign, 1 digit, 1 radix
/// point, 33 digits, 1 `E`, 1 sign and 4 exponent digits; the widest plain
/// form `-0.00000999...` is 1 sign, 1 zero, 1 radix point, 5 leading zeros and
/// 34 digits. Each form reaches exactly 42.
///
/// The plain form's five leading zeros are pinned by the GDAS rule that
/// selects it only when the adjusted exponent is at least -6, which puts five
/// zeros between the radix point and the first significant digit. The exponent
/// field cannot exceed four digits inside decimal128's range, so neither
/// number can drift without the format itself changing.
///
/// The renderers are checked against this bound once, on entry, rather than at
/// every write: the buffer is confirmed to be at least this long and the forms
/// above are what guarantee nothing longer is produced. Sizing it exactly
/// rather than generously is what keeps the two halves of that argument in
/// contact; with slack, an error in the arithmetic above would quietly consume
/// the slack instead of showing up.
///
/// A caller sizes their buffer with it, and `[u8; MAX_LEN]` puts it in the
/// type, so an undersized buffer is a compile error. No NUL is written, so no
/// room is reserved for one.
pub const MAX_LEN: usize = 42;

/// The canonical positive quiet NaN.
///
/// This is what [`bid128_parse`] and [`dpd128_parse`] write to their
/// out-parameter when a parse fails, so that a caller who ignores the returned
/// message still holds a legal IEEE 754-2019 datum rather than a stale one. It
/// is exported so a caller can name that pattern — to test for it, or to seed
/// a variable with it — and it is the same bit pattern in both formats.
pub const QNAN: u128 = (QNAN_PATTERN_HI as u128) << 64;

const QNAN_PATTERN_HI: u64 = 0x7C00_0000_0000_0000;

/* ------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/// Render a BID128 encoding into a caller-supplied buffer, allocating nothing.
///
/// The output is the GDAS `to-scientific-string` form: integer notation when the quantum
/// exponent is zero (e.g. `"123"`), plain notation when the quantum exponent is negative and
/// the adjusted exponent is at least -6 (e.g. `"123.45"`), normalized scientific notation
/// otherwise (e.g. `"1.234E+5"`). The exponent sign is always written, as in `"6.02E+23"`.
/// Cohort information is preserved: `"1.00"` and `"1.0"` are distinct outputs from distinct
/// encodings of the same numerical value.
///
/// Special values render as `"Infinity"`, `"-Infinity"`, `"NaN"` and `"sNaN"`, each with an
/// optional sign and an optional payload suffix such as `"NaN123"`.
///
/// The two interchange encodings differ only in how the coefficient is stored: BID128 as a
/// binary integer, DPD128 as a leading digit plus eleven 10-bit declets. The sign and the
/// 14-bit biased exponent are the same in both, and peer encodings of the same value render to
/// identical text.
///
/// This cannot fail. Every 128-bit pattern renders: a non-canonical BID128 coefficient (>=
/// 10^34) renders as zero at the encoded exponent, and a NaN payload >= 10^33 renders as no
/// payload, both per IEEE 754-2019 section 3.5.2.
///
/// In Rust: writes the text into `out` and returns the part written. The returned `&str`
/// borrows `out`, so nothing is allocated and nothing is copied -- it is a view of the bytes
/// just written, and `out` must outlive it. `MAX_LEN` is part of the buffer's type, so an
/// undersized buffer is a compile error rather than a runtime check.
///
/// ```
/// # use decimal128_text_codec::{MAX_LEN, bid128_to_ascii};
/// let mut buf = [0u8; MAX_LEN];
/// let text = bid128_to_ascii(&mut buf, 0x3040_0000_0000_0000_0000_0000_0000_007B);
/// assert_eq!(text, "123");
/// ```
#[must_use]
pub fn bid128_to_ascii(out: &mut [u8; MAX_LEN], bits: u128) -> &str {
    let n = decode_bid128_to_ascii(out, hi_word(bits), lo_word(bits));
    as_str(&out[..n])
}

/// Render a DPD128 encoding into a caller-supplied buffer, allocating nothing.
///
/// The output is the GDAS `to-scientific-string` form: integer notation when the quantum
/// exponent is zero (e.g. `"123"`), plain notation when the quantum exponent is negative and
/// the adjusted exponent is at least -6 (e.g. `"123.45"`), normalized scientific notation
/// otherwise (e.g. `"1.234E+5"`). The exponent sign is always written, as in `"6.02E+23"`.
/// Cohort information is preserved: `"1.00"` and `"1.0"` are distinct outputs from distinct
/// encodings of the same numerical value.
///
/// Special values render as `"Infinity"`, `"-Infinity"`, `"NaN"` and `"sNaN"`, each with an
/// optional sign and an optional payload suffix such as `"NaN123"`.
///
/// The two interchange encodings differ only in how the coefficient is stored: BID128 as a
/// binary integer, DPD128 as a leading digit plus eleven 10-bit declets. The sign and the
/// 14-bit biased exponent are the same in both, and peer encodings of the same value render to
/// identical text.
///
/// This cannot fail. Every 128-bit pattern renders: a DPD128 coefficient is always in range,
/// since a leading digit of at most 9 followed by eleven declets cannot reach 10^34; a NaN
/// payload, carried in those same declets, is likewise always in range; and non-canonical
/// declets decode to their value, per IEEE 754-2019 section 3.5.2.
///
/// In Rust: writes the text into `out` and returns the part written. The returned `&str`
/// borrows `out`, so nothing is allocated and nothing is copied -- it is a view of the bytes
/// just written, and `out` must outlive it. `MAX_LEN` is part of the buffer's type, so an
/// undersized buffer is a compile error rather than a runtime check.
///
/// ```
/// # use decimal128_text_codec::{MAX_LEN, dpd128_to_ascii};
/// let mut buf = [0u8; MAX_LEN];
/// let text = dpd128_to_ascii(&mut buf, 0x2208_0000_0000_0000_0000_0000_0000_00A3);
/// assert_eq!(text, "123");
/// ```
#[must_use]
pub fn dpd128_to_ascii(out: &mut [u8; MAX_LEN], bits: u128) -> &str {
    let n = decode_dpd128_to_ascii(out, hi_word(bits), lo_word(bits));
    as_str(&out[..n])
}

/// Parse a decimal character sequence into a BID128 encoding.
///
/// On failure the encoding written is the canonical positive quiet NaN, so a caller who
/// discards the message still observes a legal IEEE 754-2019 datum rather than stale or partial
/// data. The message is the engine's ParseStatus name for the failure -- EMPTY_STRING,
/// INVALID_FORMAT, UNEXPECTED_CHAR, DOUBLE_DOT, NO_COEFFICIENT_DIGIT, NO_EXPONENT_DIGIT,
/// INVALID_UNDERSCORE_LOCATION, FAILURE_PARSING_INFINITY or FAILURE_PARSING_NAN -- and the
/// conformance suite asserts it against the shared corpus, so it is part of the contract.
///
/// Accepted input:
///
/// - Plain notation: `"0"`, `"-1.5"`, `"123.456"`, `"0.0001"`.
/// - Scientific notation: `"1.5E10"`, `"-2.3e-7"`, `"6.02E+23"`. Both `E` and `e` are accepted
///   as the exponent separator.
/// - Digit separators: `_` between two digits, in the coefficient or the exponent (`"1_000"`,
///   `"1.234_567"`, `"1e1_0"`). An underscore must be both preceded and followed by a digit, so
///   `"1_"`, `"_1"`, `"1__2"`, `"1_.5"` and `"1._5"` are all rejected.
/// - The case-insensitive words `Infinity`, `Inf`, `NaN`, `qNaN` and `sNaN`, each with an
///   optional sign and an optional NaN payload of decimal digits (`"NaN123"`, `"-sNaN42"`). A
///   payload may also be parenthesised, as `"NaN(123)"`.
///
/// The parenthesised payload form is positional and balanced: `(` immediately after the word,
/// at least one digit, `)` last. `"NaN()"`, `"NaN(1"`, `"NaN(1)2(3)"`, `"NaN[1]"` and
/// `"NaN{1}"` are rejected. Rendering never emits parentheses, and the full decimal128 ports
/// accept the same spelling.
///
/// More than 34 significant digits round to 34, ties-to-even. Magnitudes above the
/// representable range become `Infinity`; magnitudes below it become a subnormal or zero. None
/// of those are failures.
///
/// Failures are structural: an empty sequence, unrecognized characters, missing digits around
/// the radix point, a malformed exponent field.
///
/// The two interchange encodings differ only in how the coefficient is stored: BID128 as a
/// binary integer, DPD128 as a leading digit plus eleven 10-bit declets. The same text parses
/// to peer encodings of the same value in either format, and the canonical positive quiet NaN
/// is the same bit pattern in both.
///
/// In Rust: returns `None` on success, with the encoding written to `out`, and `Some(message)`
/// on failure; `out` is written either way.
///
/// ```
/// # use decimal128_text_codec::{MAX_LEN, QNAN, bid128_parse, bid128_to_ascii};
/// let mut bits = 0u128;
/// assert_eq!(bid128_parse(&mut bits, "6.02E+23"), None);
/// let mut buf = [0u8; MAX_LEN];
/// assert_eq!(bid128_to_ascii(&mut buf, bits), "6.02E+23");
///
/// assert_eq!(bid128_parse(&mut bits, "1_"), Some("INVALID_UNDERSCORE_LOCATION"));
/// assert_eq!(bits, QNAN); // the failed parse left a legal datum behind
/// ```
pub fn bid128_parse(out: &mut u128, str: &str) -> Option<&'static str> {
    parse_into(out, true, str)
}

/// Parse a decimal character sequence into a DPD128 encoding.
///
/// On failure the encoding written is the canonical positive quiet NaN, so a caller who
/// discards the message still observes a legal IEEE 754-2019 datum rather than stale or partial
/// data. The message is the engine's ParseStatus name for the failure -- EMPTY_STRING,
/// INVALID_FORMAT, UNEXPECTED_CHAR, DOUBLE_DOT, NO_COEFFICIENT_DIGIT, NO_EXPONENT_DIGIT,
/// INVALID_UNDERSCORE_LOCATION, FAILURE_PARSING_INFINITY or FAILURE_PARSING_NAN -- and the
/// conformance suite asserts it against the shared corpus, so it is part of the contract.
///
/// Accepted input:
///
/// - Plain notation: `"0"`, `"-1.5"`, `"123.456"`, `"0.0001"`.
/// - Scientific notation: `"1.5E10"`, `"-2.3e-7"`, `"6.02E+23"`. Both `E` and `e` are accepted
///   as the exponent separator.
/// - Digit separators: `_` between two digits, in the coefficient or the exponent (`"1_000"`,
///   `"1.234_567"`, `"1e1_0"`). An underscore must be both preceded and followed by a digit, so
///   `"1_"`, `"_1"`, `"1__2"`, `"1_.5"` and `"1._5"` are all rejected.
/// - The case-insensitive words `Infinity`, `Inf`, `NaN`, `qNaN` and `sNaN`, each with an
///   optional sign and an optional NaN payload of decimal digits (`"NaN123"`, `"-sNaN42"`). A
///   payload may also be parenthesised, as `"NaN(123)"`.
///
/// The parenthesised payload form is positional and balanced: `(` immediately after the word,
/// at least one digit, `)` last. `"NaN()"`, `"NaN(1"`, `"NaN(1)2(3)"`, `"NaN[1]"` and
/// `"NaN{1}"` are rejected. Rendering never emits parentheses, and the full decimal128 ports
/// accept the same spelling.
///
/// More than 34 significant digits round to 34, ties-to-even. Magnitudes above the
/// representable range become `Infinity`; magnitudes below it become a subnormal or zero. None
/// of those are failures.
///
/// Failures are structural: an empty sequence, unrecognized characters, missing digits around
/// the radix point, a malformed exponent field.
///
/// The two interchange encodings differ only in how the coefficient is stored: BID128 as a
/// binary integer, DPD128 as a leading digit plus eleven 10-bit declets. The same text parses
/// to peer encodings of the same value in either format, and the canonical positive quiet NaN
/// is the same bit pattern in both.
///
/// In Rust: returns `None` on success, with the encoding written to `out`, and `Some(message)`
/// on failure; `out` is written either way.
///
/// ```
/// # use decimal128_text_codec::{MAX_LEN, QNAN, dpd128_parse, dpd128_to_ascii};
/// let mut bits = 0u128;
/// assert_eq!(dpd128_parse(&mut bits, "6.02E+23"), None);
/// let mut buf = [0u8; MAX_LEN];
/// assert_eq!(dpd128_to_ascii(&mut buf, bits), "6.02E+23");
///
/// assert_eq!(dpd128_parse(&mut bits, "1_"), Some("INVALID_UNDERSCORE_LOCATION"));
/// assert_eq!(bits, QNAN); // the failed parse left a legal datum behind
/// ```
pub fn dpd128_parse(out: &mut u128, str: &str) -> Option<&'static str> {
    parse_into(out, false, str)
}

/// View the written prefix of a render buffer as text.
///
/// The renderers write ASCII only, so this cannot fail; the check stands where
/// `String::from_utf8` stood when the renderers returned an owned string, and
/// keeps the crate free of `unsafe`.
fn as_str(bytes: &[u8]) -> &str {
    if let Ok(s) = core::str::from_utf8(bytes) {
        return s;
    }
    impossible("the renderer wrote a non-ASCII byte")
}

/// The encoding is written on both paths -- the encoding on success, the canonical positive
/// quiet NaN on failure -- so the caller never reads back what was there before the call. That
/// is why a failure is reported beside the value rather than in place of it: a caller who
/// discards the message still holds a legal IEEE 754-2019 datum.
fn parse_into(out: &mut u128, format_is_bid: bool, str: &str) -> Option<&'static str> {
    let result = encode_string_or_error(format_is_bid, str.as_bytes());
    if let Ok((hi, lo)) = result {
        *out = join_words(hi, lo);
        return None;
    }
    *out = QNAN;
    result.err()
}

/* ------------------------------------------------------------------------
 * Format constants -- IEEE 754-2019 Table 3.6, decimal128
 * ---------------------------------------------------------------------- */

const BIAS: i32 = 6176;

/// The minimum quantum exponent.
const Q_MIN: i32 = -6176;

/// The maximum quantum exponent.
const Q_MAX: i32 = 6111;

/// Trailing significand field width in bits.
const T_WIDTH: u32 = 110;

/// Combination field width in bits.
const W5: u32 = 17;

/// 10^34, one past the largest coefficient; maxx = max eXclusive.
const MAXX_COEFF_34_HI: u64 = 0x0001_ED09_BEAD_87C0;

const MAXX_COEFF_34_LO: u64 = 0x378D_8E64_0000_0000;

/// 10^33, the smallest 34-digit coefficient.
const MIN_PRECISION_34_COEFF_HI: u64 = 0x0000_314D_C644_8D93;

const MIN_PRECISION_34_COEFF_LO: u64 = 0x38C1_5B0A_0000_0000;

/// 33 nines -- the largest canonical NaN payload.
const MAX_PAYLOAD_HI: u64 = MIN_PRECISION_34_COEFF_HI;

const MAX_PAYLOAD_LO: u64 = MIN_PRECISION_34_COEFF_LO - 1;

const SIGN_BIT: u64 = 0x8000_0000_0000_0000;

const INF_PATTERN_HI: u64 = 0x7800_0000_0000_0000;

const SNAN_PATTERN_HI: u64 = 0x7E00_0000_0000_0000;

const TEN_POW_15: u64 = 1_000_000_000_000_000;

const TEN_POW_18: u64 = 1_000_000_000_000_000_000;

/*
 * Powers of 10 from 10^0 through 10^34 as 128-bit values in little-endian
 * order: POW10[2*i] is the low 64 bits of 10^i, POW10[2*i + 1] the high 64.
 *
 * Sized to the coefficient bound and indexed directly -- no mask, no padding.
 * Every reachable index is <= 34: the binding case is calc_digit_len_128, whose
 * input is always a coefficient or NaN payload below 10^34, so bit_len <= 113
 * and (bit_len * 1233) >> 12 <= 34. It doubles as the 5^n source for the divide
 * kernel, since 10^n >> n is exactly 5^n; there is no POW5 table.
 */
#[rustfmt::skip]
static POW10: [u64; (34 + 1) * 2] = [
    1, 0, // 10^0
    0x0000_0000_0000_000A, 0, // 10^1
    0x0000_0000_0000_0064, 0, // 10^2
    0x0000_0000_0000_03E8, 0, // 10^3
    0x0000_0000_0000_2710, 0, // 10^4
    0x0000_0000_0001_86A0, 0, // 10^5
    0x0000_0000_000F_4240, 0, // 10^6
    0x0000_0000_0098_9680, 0, // 10^7
    0x0000_0000_05F5_E100, 0, // 10^8
    0x0000_0000_3B9A_CA00, 0, // 10^9
    0x0000_0002_540B_E400, 0, // 10^10
    0x0000_0017_4876_E800, 0, // 10^11
    0x0000_00E8_D4A5_1000, 0, // 10^12
    0x0000_0918_4E72_A000, 0, // 10^13
    0x0000_5AF3_107A_4000, 0, // 10^14
    0x0003_8D7E_A4C6_8000, 0, // 10^15
    0x0023_86F2_6FC1_0000, 0, // 10^16
    0x0163_4578_5D8A_0000, 0, // 10^17
    0x0DE0_B6B3_A764_0000, 0, // 10^18
    0x8AC7_2304_89E8_0000, 0, // 10^19
    0x6BC7_5E2D_6310_0000, 0x0000_0000_0000_0005, // 10^20
    0x35C9_ADC5_DEA0_0000, 0x0000_0000_0000_0036, // 10^21
    0x19E0_C9BA_B240_0000, 0x0000_0000_0000_021E, // 10^22
    0x02C7_E14A_F680_0000, 0x0000_0000_0000_152D, // 10^23
    0x1BCE_CCED_A100_0000, 0x0000_0000_0000_D3C2, // 10^24
    0x1614_0148_4A00_0000, 0x0000_0000_0008_4595, // 10^25
    0xDCC8_0CD2_E400_0000, 0x0000_0000_0052_B7D2, // 10^26
    0x9FD0_803C_E800_0000, 0x0000_0000_033B_2E3C, // 10^27
    0x3E25_0261_1000_0000, 0x0000_0000_204F_CE5E, // 10^28
    0x6D72_17CA_A000_0000, 0x0000_0001_431E_0FAE, // 10^29
    0x4674_EDEA_4000_0000, 0x0000_000C_9F2C_9CD0, // 10^30
    0xC091_4B26_8000_0000, 0x0000_007E_37BE_2022, // 10^31
    0x85AC_EF81_0000_0000, 0x0000_04EE_2D6D_415B, // 10^32
    0x38C1_5B0A_0000_0000, 0x0000_314D_C644_8D93, // 10^33
    0x378D_8E64_0000_0000, 0x0001_ED09_BEAD_87C0, // 10^34
];

const EXACT: i32 = 0;

const LT_HALF: i32 = 1;

/// `HALF` alone would read as a quantity.
const TIE_HALF: i32 = 2;

const GT_HALF: i32 = 3;

/// Two bits per digit 0..=9: 0 -> EXACT, 1..=4 -> LT_HALF, 5 -> TIE_HALF, 6..=9 -> GT_HALF
const DIGIT_MAP: u64 = 0xFF954;

/*
 * Reciprocal-multiplication constants. The 64-bit ones fold in the 64-bit shift
 * that mulhi64 already performs, so the named shift is the remainder.
 */
const M_U64_DIV_1E8: u64 = 0x8000_0000_0000_0000 | 0x2BCC_7711_8461_CEFD;

const S_U64_DIV_1E8: u32 = 26;

const M_U32_DIV_1E1: u64 = 0xCCCC_CCCD;

const S_U32_DIV_1E1: u32 = 35;

const M_U32_DIV_1E2: u64 = 0x51EB_851F;

const S_U32_DIV_1E2: u32 = 37;

const M_U64_DIV_1E4: u64 = 0x346D_C5D6_3886_594B;

const S_U64_DIV_1E4: u32 = 11;

const MASK32: u64 = 0xFFFF_FFFF;

/// None of the optional leading characters live in the string data. The sign,
/// and the `s` that marks a signaling NaN, are each written unconditionally and
/// then overwritten by whatever follows when they are not wanted -- the trick
/// [`fnz_to_integer_ascii`] already uses for the sign. That holds only because
/// every optional character is a single byte at the front of the form: a
/// spelling that ever put one elsewhere would render wrong rather than fail to
/// compile. The one exception is written where it happens, in [`zer_ascii`]'s
/// exponent form, where the byte is a TRAILING `+` overwritten by
/// [`append_int`]'s own `-`.
const INFINITY_TEXT: &[u8] = b"Infinity";

const NAN_TEXT: &[u8] = b"NaN";

/// Every plain-form zero is a prefix of this literal. `k = -q_exp` is the count
/// of zeros after the radix point, so `k = 0` renders `"0"` with no radix point
/// at all and `k >= 1` renders `"0."` followed by `k` zeros -- lengths 1 and
/// `k + 2`.
///
/// The sign is not in the data. It is written into `out[0]` unconditionally and
/// overwritten by the `0` when the value is positive, the same trick the finite
/// renderers use. That works because the negative spelling of every form here is
/// the positive spelling with one byte in front; a form that ever put the sign
/// anywhere else would render wrong rather than fail to compile.
const PLAIN_ZERO: &[u8] = b"0.000000";

/// The exponent form's base. The `+` is not here: see [`zer_ascii`] below.
const ZERO_EXP_TEXT: &[u8] = b"0E";

/*
 * The parse failures, one message each.
 *
 * A failure crosses the API as the message itself. The text is the engine's
 * ParseStatus name for the failure and the conformance suite asserts it against
 * the shared corpus, so the wording is the contract rather than an incidental
 * choice.
 *
 * These are `&'static str` rather than a dedicated error type on purpose. An
 * error type would be the more idiomatic surface, but what the corpus pins down
 * is the ParseStatus name, and wrapping it puts a layer between the contract
 * and the thing that states it. Sharing one constant per failure is also what
 * lets the two formats' failures be compared directly: the same input must be
 * rejected for the same reason in both, and that check is `==` on two `&str`.
 */
const ERR_EMPTY_STRING: &str = "EMPTY_STRING";

const ERR_INVALID_FORMAT: &str = "INVALID_FORMAT";

const ERR_UNEXPECTED_CHAR: &str = "UNEXPECTED_CHAR";

const ERR_DOUBLE_RADIX_DOT: &str = "DOUBLE_DOT";

const ERR_NO_COEFFICIENT_DIGIT: &str = "NO_COEFFICIENT_DIGIT";

const ERR_NO_EXPONENT_DIGIT: &str = "NO_EXPONENT_DIGIT";

const ERR_UNDERSCORE_LOCATION: &str = "INVALID_UNDERSCORE_LOCATION";

const ERR_PARSING_INFINITY: &str = "FAILURE_PARSING_INFINITY";

const ERR_PARSING_NAN: &str = "FAILURE_PARSING_NAN";

/* --------------------------------------------------------------------------
 * The 128-bit boundary
 *
 * Public in, public out. Everything between these and the encoders works in the
 * reference's `dw1` / `dw0` word pair.
 * ------------------------------------------------------------------------ */

#[inline(always)]
fn hi_word(bits: u128) -> u64 {
    (bits >> 64) as u64
}

#[inline(always)]
fn lo_word(bits: u128) -> u64 {
    bits as u64
}

#[inline(always)]
fn join_words(hi: u64, lo: u64) -> u128 {
    ((hi as u128) << 64) | (lo as u128)
}

/* ------------------------------------------------------------------------
 * Low-level unsigned arithmetic
 * ---------------------------------------------------------------------- */

/// Compute the high 64 bits of the 128-bit unsigned product `x * y`.
#[inline(always)]
fn mulhi64(x: u64, y: u64) -> u64 {
    (((x as u128) * (y as u128)) >> 64) as u64
}

/// Compute the bit-length of a 128-bit unsigned value: the position of the highest set bit,
/// plus one. Returns 0 for the zero value.
fn calc_bit_len_128(dw1: u64, dw0: u64) -> u32 {
    if dw1 != 0 {
        return 128 - dw1.leading_zeros();
    }
    if dw0 != 0 {
        return 64 - dw0.leading_zeros();
    }
    0
}

/// Compute the decimal digit count of a 128-bit unsigned value. Uses the standard log10/log2
/// approximation: for any non-zero `bit_len`, `(bit_len * 1233) >> 12` is `floor(bit_len *
/// log10(2))`, which is either the exact digit count or one too small. A single unsigned
/// 128-bit compare against the next-larger power of 10 in `POW10` resolves the off-by-one.
fn calc_digit_len_128(dw1: u64, dw0: u64) -> usize {
    let bit_len = calc_bit_len_128(dw1, dw0) as usize;
    let lo_digit_count = (bit_len * 1233) >> 12;
    let hi_digit_count = lo_digit_count + 1;
    let pow10_offset = lo_digit_count << 1;
    let p1 = POW10[pow10_offset + 1];
    let p0 = POW10[pow10_offset];
    if dw1 != p1 {
        return hi_digit_count - usize::from(dw1 < p1);
    }
    hi_digit_count - usize::from(dw0 < p0)
}

/// Multiply a 64-bit unsigned value `x` by `10^pow10` and add a 64-bit integer `a`, producing
/// the 128-bit result.
fn u128_fma_pow10(x: u64, pow10: usize, a: u64) -> (u64, u64) {
    let y = POW10[pow10 << 1];
    // This product is relied on to wrap past 64 bits, and recovers
    // the high half separately; taking it in u128 says the same thing directly.
    let prod = (x as u128) * (y as u128);
    let prod_hi = (prod >> 64) as u64;
    let prod_lo = prod as u64;
    let (sum_lo, carry) = prod_lo.overflowing_add(a);
    (prod_hi + u64::from(carry), sum_lo)
}

/* ------------------------------------------------------------------------
 * The divide kernel
 * ---------------------------------------------------------------------- */

/// Divide a 113-bit unsigned coefficient by `10^pow10` for `pow10` in 1..9, producing the
/// quotient and the remainder. Factors `10^pow10` as `2^pow10 * 5^pow10`: the low `pow10` bits
/// are extracted directly as the low part of the remainder, then the shifted value is divided
/// by `5^pow10` using two hardware 64-bit divides.
fn u113_div_mod_pow10_max9(dw_hi: u64, dw_lo: u64, pow10: u32) -> (u64, u64, u64) {
    verify!((1..=9).contains(&pow10));
    verify!(dw_hi.leading_zeros() >= 15);

    let mask = (1u64 << pow10) - 1;
    let b = dw_lo & mask;
    let s_hi = dw_hi >> pow10;
    let s_lo = (dw_lo >> pow10) | (dw_hi << (64 - pow10));

    let k = 50 - pow10;
    let mask_k = (1u64 << k) - 1;
    let hi63 = (s_hi << (14 + pow10)) | (s_lo >> k);
    let lo_k = s_lo & mask_k;

    // two slots per power in POW10, hence the doubled index
    let d = POW10[(pow10 << 1) as usize] >> pow10; // = 5^pow10
    let q1 = hi63 / d;
    let r1 = hi63 % d;
    let mid = (r1 << k) | lo_k;
    let q0 = mid / d;
    let r0 = mid % d;

    (q1 >> (14 + pow10), (q1 << k) + q0, (r0 << pow10) | b)
}

/* ------------------------------------------------------------------------
 * Residue -- how much was discarded, relative to half a ulp
 * ---------------------------------------------------------------------- */

/// Classify a single discarded decimal digit (0..9) as `EXACT`, `LT_HALF`, `HALF`, or `GT_HALF`
/// via lookup in `DIGIT_MAP`.
fn residue_from_decimal_digit(digit: u32) -> i32 {
    ((DIGIT_MAP >> (digit << 1)) & 0x03) as i32
}

/// Combine a primary `round_residue` with a `sticky_residue` from less-significant digits, in
/// the round-bit-and-sticky-bit style. The sticky residue is collapsed to a single "nonzero"
/// bit and OR'd into the low bit of the round residue.
fn residue_merge(round_residue: i32, sticky_residue: i32) -> i32 {
    let s = (sticky_residue & 1) | (sticky_residue >> 1);
    (round_residue | s) & 0x03
}

/// Classify a 128-bit unsigned value relative to half of `10^pow10`, returning one of `EXACT`,
/// `LT_HALF`, `HALF`, or `GT_HALF`. Used in the underflow boundary case where the entire
/// coefficient is being discarded by truncation: the residue is determined by where the
/// coefficient falls relative to half-of-the-truncation-divisor. Looks up `10^pow10` in the
/// `POW10` table, halves it via a 128-bit shift right by 1, and compares against the input as
/// an unsigned 128-bit value.
fn residue_from_value_pow10(dw1: u64, dw0: u64, pow10: usize) -> i32 {
    if (dw1 | dw0) == 0 {
        return EXACT;
    }
    let pow10_index = pow10 << 1;
    let pow10_hi = POW10[pow10_index + 1];
    let pow10_lo = POW10[pow10_index];

    let half_hi = pow10_hi >> 1;
    let half_lo = (pow10_hi << 63) | (pow10_lo >> 1);
    if dw1 != half_hi {
        return if dw1 < half_hi { LT_HALF } else { GT_HALF };
    }
    if dw0 != half_lo {
        return if dw0 < half_lo { LT_HALF } else { GT_HALF };
    }
    TIE_HALF
}

/// Divide a 128-bit unsigned coefficient by `10^truncation_needed`, producing the quotient and
/// the round-and-sticky residue class of what was discarded. Truncates in chunks of up to 9
/// digits via `u113_div_mod_pow10_max9`.
fn u128_scale_down_pow10(dw1: u64, dw0: u64, truncation_needed: usize) -> (u64, u64, i32) {
    verify!((dw1 | dw0) != 0);
    verify!(truncation_needed > 0 && truncation_needed < calc_digit_len_128(dw1, dw0));
    let mut scale_residue = EXACT;
    let mut dw1_t = dw1;
    let mut dw0_t = dw0;
    let mut digits_remaining = truncation_needed;
    while digits_remaining > 0 {
        let step_pow10 = digits_remaining.min(9);
        let rem;
        (dw1_t, dw0_t, rem) = u113_div_mod_pow10_max9(dw1_t, dw0_t, step_pow10 as u32);
        let step_residue = residue_from_value_pow10(0, rem, step_pow10);
        scale_residue = residue_merge(step_residue, scale_residue);
        digits_remaining -= step_pow10;
    }
    (dw1_t, dw0_t, scale_residue)
}

/* --------------------------------------------------------------------------
 * DPD declet conversion
 *
 * A declet is 10 bits holding three decimal digits, 0..=999. This is the one
 * place BID and DPD genuinely differ: BID stores the 34-digit coefficient as a
 * binary integer, DPD as a leading digit plus eleven declets.
 *
 * Both directions are pure boolean arithmetic (Cowlishaw's dpd2bcd and bcd2dpd)
 * rather than lookup tables.
 * ------------------------------------------------------------------------ */

/// Decode a 10-bit DPD declet (bits 9..0) into 0..999. Mapping: bit9..bit0 as p q r s t u v w x
/// y. Accepts the 24 non-canonical declets as well as the 1000 canonical ones, per Cowlishaw's
/// dpd2bcd.
fn dpd_decode_declet(declet: u64) -> u64 {
    // A complement of t is spelled & !t here: on operands already reduced to 0 or 1 it agrees
    // with !, and the complemented bits are masked away either way.
    verify!(declet <= 1023);

    if declet == 0 {
        return 0;
    }

    let p = (declet >> 9) & 1;
    let q = (declet >> 8) & 1;
    let r = (declet >> 7) & 1;
    let s = (declet >> 6) & 1;
    let t = (declet >> 5) & 1;
    let u = (declet >> 4) & 1;
    let v = (declet >> 3) & 1;
    let w = (declet >> 2) & 1;
    let x = (declet >> 1) & 1;
    let y = declet & 1;

    let ns = s ^ 1;
    let nx = x ^ 1;
    let nv = v ^ 1;
    let nw = w ^ 1;
    let nt = t ^ 1;

    let a = (v & w) & (ns | t | nx);
    let b = p & (nv | nw | (s & !t & x));
    let c = q & (nv | nw | (s & !t & x));
    let d = r;

    let e = v & ((nw & x) | (!t & x) | (s & x));
    let f = (s & (nv | nx)) | (p & ns & t & v & w & x);
    let g = (t & (nv | nx)) | (q & ns & t & w);
    let h = u;

    let i = v & ((nw & nx) | (w & x & (s | t)));
    let j = (nv & w) | (s & v & nw & x) | (p & w & (nx | (ns & !t)));
    let k = (nv & x) | (t & nw & x) | (q & v & w & (nx | (ns & nt)));
    let m = y;

    let d2 = (a << 3) | (b << 2) | (c << 1) | d;
    let d1 = (e << 3) | (f << 2) | (g << 1) | h;
    let d0 = (i << 3) | (j << 2) | (k << 1) | m;

    d2 * 100 + d1 * 10 + d0
}

/// Encode 0..999 into a 10-bit DPD declet, per Cowlishaw's bcd2dpd. The inverse of
/// `dpd_decode_declet` over canonical declets. Non-canonical declets decode to a value that
/// re-encodes to the canonical declet for that value, so `encode(decode(x)) == x` only for
/// canonical x.
fn dpd_encode_declet(n: u64) -> u64 {
    verify!(n <= 999);

    if n == 0 {
        return 0;
    }

    let q10 = (n * 205) >> 11; // n / 10, exact for n <= 999
    let q100 = (n * 41) >> 12; // n / 100, exact for n <= 999

    let d2 = q100;
    let d1 = q10 - (q100 * 10);
    let d0 = n - (q10 * 10);

    let a = (d2 >> 3) & 1;
    let b = (d2 >> 2) & 1;
    let c = (d2 >> 1) & 1;
    let d = d2 & 1;

    let e = (d1 >> 3) & 1;
    let f = (d1 >> 2) & 1;
    let g = (d1 >> 1) & 1;
    let h = d1 & 1;

    let i = (d0 >> 3) & 1;
    let j = (d0 >> 2) & 1;
    let k = (d0 >> 1) & 1;
    let m = d0 & 1;

    let na = a ^ 1;
    let ne = e ^ 1;
    let ni = i ^ 1;

    let p = b | ((a & j) | ((a & f) & i));
    let q = c | ((a & k) | ((a & g) & i));
    let r = d;
    let s = (f & (na | ni)) | ((na & (e & j)) | (e & i));
    let t = g | ((na & (e & k)) | (a & i));
    let u = h;
    let v = a | (e | i);
    let w = a | ((e & i) | (ne & j));
    let x = e | ((a & i) | (na & k));
    let y = m;

    (p << 9)
        | (q << 8)
        | (r << 7)
        | (s << 6)
        | (t << 5)
        | (u << 4)
        | (v << 3)
        | (w << 2)
        | (x << 1)
        | y
}

/// Decode up to seven packed DPD declets into a binary integer. Declets are ten bits each,
/// least significant first. The topmost may be narrower than ten bits, which is how the
/// coefficient's leading digit is carried: 0..9 encodes to itself as a declet.
fn bin_from_declets(declets: u64) -> u64 {
    let mut bin: u64 = 0;
    let mut rest = declets;
    let mut multiplier: u64 = 1;
    while rest != 0 {
        bin += dpd_decode_declet(rest & 0x3FF) * multiplier;
        // The last iteration multiplies past 10^18 and the product is never
        // used, since `rest` is zero by then, so it is let wrap.
        multiplier = multiplier.wrapping_mul(1000);
        rest >>= 10;
    }
    bin
}

/// Encode a binary integer of at most 18 digits into six packed declets, least significant
/// first. The inverse of `bin_from_declets` over that range.
fn declets_from_bin(bin: u64) -> u64 {
    verify!(bin < TEN_POW_18);
    let mut rest = bin;
    let mut declets: u64 = 0;
    let mut shift = 0;
    while rest != 0 {
        let q = mulhi64(rest, 0x020C_49BA_5E35_3F7D) >> 3; // rest / 1000
        declets |= dpd_encode_declet(rest - (q * 1000)) << shift;
        shift += 10;
        rest = q;
    }
    declets
}

/* ------------------------------------------------------------------------
 * Rendering integers into ASCII
 * ---------------------------------------------------------------------- */

/// Render exactly 8 decimal digits of `dw` into `ascii`, ending at `off_maxx`. Caller must
/// ensure `dw < 10^8`; the function pads with leading zeros if `dw` has fewer than 8
/// significant digits. Implemented as a binary cascade: / 10^4 once to split into two 4-digit
/// halves, then / 10^2 twice to split each half into two 2-digit pairs, then / 10 four times to
/// extract individual digits. Always writes exactly 8 bytes at offsets `off_maxx - 8` through
/// `off_maxx - 1`.
fn render8_digits_before_index(dw: u64, ascii: &mut [u8], off_maxx: usize) {
    let abcd = mulhi64(dw, M_U64_DIV_1E4) >> S_U64_DIV_1E4;
    let efgh = dw - (abcd * 10000);

    let ab = (abcd * M_U32_DIV_1E2) >> S_U32_DIV_1E2;
    let cd = abcd - (ab * 100);

    let ef = (efgh * M_U32_DIV_1E2) >> S_U32_DIV_1E2;
    let gh = efgh - (ef * 100);

    let a = (ab * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
    let b = ab - (a * 10);

    let c = (cd * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
    let d = cd - (c * 10);

    let e = (ef * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
    let f = ef - (e * 10);

    let g = (gh * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
    let h = gh - (g * 10);

    demand(
        off_maxx >= 8 && off_maxx <= ascii.len(),
        "render8DigitsBeforeIndex: offset out of range",
    );
    ascii[off_maxx - 8] = b'0' + a as u8;
    ascii[off_maxx - 7] = b'0' + b as u8;
    ascii[off_maxx - 6] = b'0' + c as u8;
    ascii[off_maxx - 5] = b'0' + d as u8;
    ascii[off_maxx - 4] = b'0' + e as u8;
    ascii[off_maxx - 3] = b'0' + f as u8;
    ascii[off_maxx - 2] = b'0' + g as u8;
    ascii[off_maxx - 1] = b'0' + h as u8;
}

/// Render exactly `render_digit_count` decimal digits of `dw` into `ascii`, ending at
/// `off_maxx`. Pads with leading zeros if `dw` has fewer significant digits than
/// `render_digit_count`. Two phases:
///
/// - While at least 4 digits remain: peel 4 digits per iteration via / 10^4 followed by a
///   2-digit / 1-digit cascade.
/// - For the remaining 0..3 digits: peel one digit at a time via / 10 through reciprocal
///   multiplication.
///
/// Always writes exactly `render_digit_count` bytes at offsets `off_maxx - render_digit_count`
/// through `off_maxx - 1`.
fn render_tail_digits_before_index(
    render_digit_count: usize,
    dw: u64,
    ascii: &mut [u8],
    off_maxx: usize,
) {
    let mut t = dw;
    let mut remaining_digit_count = render_digit_count;
    let mut ib = off_maxx;
    while remaining_digit_count >= 4 {
        let t0 = mulhi64(t, M_U64_DIV_1E4) >> S_U64_DIV_1E4;
        let abcd = t - (t0 * 10000);
        t = t0;
        let ab = (abcd * M_U32_DIV_1E2) >> S_U32_DIV_1E2;
        let cd = abcd - (ab * 100);
        let a = (ab * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
        let b = ab - (a * 10);
        let c = (cd * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
        let d = cd - (c * 10);
        demand(
            ib >= 4 && ib <= ascii.len(),
            "u128ToAscii: offset out of range",
        );
        ascii[ib - 4] = b'0' + a as u8;
        ascii[ib - 3] = b'0' + b as u8;
        ascii[ib - 2] = b'0' + c as u8;
        ascii[ib - 1] = b'0' + d as u8;
        ib -= 4;
        remaining_digit_count -= 4;
    }
    while remaining_digit_count > 0 {
        let div_ten = (t * M_U32_DIV_1E1) >> S_U32_DIV_1E1;
        let digit = t - (div_ten * 10);
        ib -= 1;
        ascii[ib] = b'0' + digit as u8;
        t = div_ten;
        remaining_digit_count -= 1;
    }
    verify!(off_maxx - ib == render_digit_count);
}

/// Render a 128-bit unsigned integer as `digit_print_count` decimal digits into `ascii`, ending
/// at offset `off + digit_print_count`. Three phases:
///
/// - While `dw1_t != 0`: peel 8 digits per outer iteration via three / 10^8 divisions on
///   rotating 96-bit chunks, using `mulhi64` for the high half of each multi-word divide.
/// - Once the value fits in 64 bits: peel 8 digits at a time via a single / 10^8 per iteration.
/// - For 1..7 leftover digits: delegate to `render_tail_digits_before_index`.
///
/// Each 8-digit chunk is rendered by `render8_digits_before_index`. The caller must have
/// allocated a buffer with at least `off + digit_print_count` bytes, and is responsible for any
/// leading-zero padding outside the rendered range.
fn u128_to_ascii(digit_print_count: usize, dw1: u64, dw0: u64, ascii: &mut [u8], off: usize) {
    let mut dw1_t = dw1;
    let mut dw0_t = dw0;
    let mut ich = off + digit_print_count;
    let mut remaining_digit_count = digit_print_count;
    while dw1_t != 0 {
        let q2 = mulhi64(dw1_t, M_U64_DIV_1E8) >> S_U64_DIV_1E8;
        let r2 = dw1_t - (q2 * 100_000_000);
        let s2 = (r2 << 32) | (dw0_t >> 32);
        let q1 = mulhi64(s2, M_U64_DIV_1E8) >> S_U64_DIV_1E8;
        let r1 = s2 - (q1 * 100_000_000);
        let s1 = (r1 << 32) | (dw0_t & MASK32);
        let q0 = mulhi64(s1, M_U64_DIV_1E8) >> S_U64_DIV_1E8;
        let r0 = s1 - (q0 * 100_000_000);
        dw0_t = (q1 << 32) + q0;
        dw1_t = q2;
        render8_digits_before_index(r0, ascii, ich);
        ich -= 8;
        remaining_digit_count -= 8;
    }
    while remaining_digit_count >= 8 {
        let t0 = mulhi64(dw0_t, M_U64_DIV_1E8) >> S_U64_DIV_1E8;
        let r0 = dw0_t - (t0 * 100_000_000);
        dw0_t = t0;
        render8_digits_before_index(r0, ascii, ich);
        ich -= 8;
        remaining_digit_count -= 8;
    }
    if remaining_digit_count > 0 {
        render_tail_digits_before_index(remaining_digit_count, dw0_t, ascii, ich);
    }
}

/// Write a signed decimal exponent at `off`, returning its length. The magnitude never exceeds
/// 6176, so it is at most four digits.
fn append_int(out: &mut [u8], off: usize, v: i32) -> usize {
    // core has no integer-to-bytes helper, and routing this through fmt would pull the
    // formatting machinery into the render path for one call site.
    let mut sign_len = 0;
    let mut magnitude = v;
    if v < 0 {
        out[off] = b'-';
        sign_len = 1;
        magnitude = -v;
    }
    let digit_len = if magnitude >= 1000 {
        4
    } else if magnitude >= 100 {
        3
    } else if magnitude >= 10 {
        2
    } else {
        1
    };
    render_tail_digits_before_index(digit_len, magnitude as u64, out, off + sign_len + digit_len);
    sign_len + digit_len
}

/* --------------------------------------------------------------------------
 * Rendering a decoded value
 *
 * These four know nothing about BID or DPD: both decoders reduce an encoding to
 * (sign, qExp, coefficient) and hand it here.
 * ------------------------------------------------------------------------ */

fn inf_ascii(out: &mut [u8], sign: bool) -> usize {
    let sign_len = usize::from(sign);
    let total_len = sign_len + INFINITY_TEXT.len();
    out[0] = b'-'; // overwritten by the b'I' when unsigned
    out[sign_len..total_len].copy_from_slice(INFINITY_TEXT);
    total_len
}

fn nan_ascii(
    out: &mut [u8],
    sign: bool,
    is_signaling: bool,
    payload_hi: u64,
    payload_lo: u64,
) -> usize {
    let sign_len = usize::from(sign);
    let s_len = usize::from(is_signaling);
    let word_off = sign_len + s_len;
    let base_len = word_off + NAN_TEXT.len();
    out[0] = b'-'; // overwritten by the b's' or the b'N' when unsigned
    out[sign_len] = b's'; // overwritten by the b'N' when quiet

    if (payload_hi | payload_lo) == 0
        // non-canonical payload, IEEE 754-2019 3.5.2 c) 2) -- read as zero
        || payload_hi > MAX_PAYLOAD_HI
        || (payload_hi == MAX_PAYLOAD_HI && payload_lo > MAX_PAYLOAD_LO)
    {
        out[word_off..base_len].copy_from_slice(NAN_TEXT);
        return base_len;
    }
    let digit_len = calc_digit_len_128(payload_hi, payload_lo);
    let total_len = base_len + digit_len;
    out[word_off..base_len].copy_from_slice(NAN_TEXT);
    u128_to_ascii(digit_len, payload_hi, payload_lo, out, base_len);
    total_len
}

fn zer_ascii(out: &mut [u8], sign: bool, q_exp: i32) -> usize {
    if (-6..=0).contains(&q_exp) {
        let k = (-q_exp) as usize;
        let zero_len = if k == 0 { 1 } else { k + 2 };
        let sign_len = usize::from(sign);
        let total_len = sign_len + zero_len;
        out[0] = b'-'; // overwritten by the b'0' when unsigned
        out[sign_len..total_len].copy_from_slice(&PLAIN_ZERO[..zero_len]);
        return total_len;
    }
    let sign_len = usize::from(sign);
    let word_len = ZERO_EXP_TEXT.len();
    // A positive exponent needs an explicit `+`; a negative one prints its own
    // `-` from append_int. Write the `+` unconditionally and let append_int land
    // on top of it when q_exp < 0 -- the same discipline as the leading sign,
    // pointed the other way. It is correct only because append_int emits a `-`
    // for a negative value and emits nothing for a positive one, and because the
    // plain form above has already taken every q_exp in [-6, 0], so this branch
    // never sees zero.
    let plus_len = usize::from(q_exp > 0);
    let base_len = sign_len + word_len + plus_len;
    out[0] = b'-'; // overwritten by the b'0' when unsigned
    out[sign_len..sign_len + word_len].copy_from_slice(ZERO_EXP_TEXT);
    out[sign_len + word_len] = b'+'; // overwritten by append_int's b'-' when q_exp < 0
    base_len + append_int(out, base_len, q_exp)
}

/// Format a finite non-zero value with qExp == 0 as a plain integer sequence: no radix point,
/// no exponent.
fn fnz_to_integer_ascii(out: &mut [u8], sign: bool, digit_len: usize, dw1: u64, dw0: u64) -> usize {
    verify!(digit_len > 0);
    let sign_len = usize::from(sign);
    let total_len = sign_len + digit_len;
    out[0] = b'-'; // overwritten by the first digit when unsigned
    u128_to_ascii(digit_len, dw1, dw0, out, sign_len);
    total_len
}

/// Format a finite non-zero value in fixed-point notation, with an explicit radix point and the
/// appropriate number of leading zeros -- "0.00012345". Renders the coefficient digits, then
/// shifts the trailing digits right by one position to open a slot for the radix point. Leading
/// zeros are written directly into the buffer before the digits.
fn fnz_to_decimal_point_ascii(
    out: &mut [u8],
    sign: bool,
    q_exp: i32,
    digit_len: usize,
    dw1: u64,
    dw0: u64,
) -> usize {
    let digits_right_of_decimal = (-q_exp) as usize;
    let leading_zero_count =
        (1 + digits_right_of_decimal as i32 - digit_len as i32).max(0) as usize;
    let sign_len = usize::from(sign);
    let decimal_point_len = 1;
    let total_len = sign_len + leading_zero_count + decimal_point_len + digit_len;
    out[0] = b'-';
    let mut i = sign_len;
    while i <= leading_zero_count {
        // one extra, overwritten below
        out[i] = b'0';
        i += 1;
    }
    u128_to_ascii(digit_len, dw1, dw0, out, sign_len + leading_zero_count);
    let mut j = total_len - 1;
    while j >= total_len - digits_right_of_decimal {
        out[j] = out[j - 1];
        j -= 1;
    }
    out[total_len - digits_right_of_decimal - 1] = b'.';
    total_len
}

/// Format a finite non-zero value in normalized scientific notation -- "1.234E+5",
/// "-9.99E-200". Renders the coefficient digits, inserts a radix point after the leading digit
/// when there is more than one, then appends E, the exponent sign and the absolute exponent.
/// The sign is always written: + for non-negative exponents, - for negative, per the GDAS
/// to-scientific-string form.
fn fnz_to_normalized_scientific_ascii(
    out: &mut [u8],
    sign: bool,
    q_exp: i32,
    digit_len: usize,
    dw1: u64,
    dw0: u64,
) -> usize {
    verify!(digit_len > 0);
    let e_exp = q_exp + (digit_len as i32 - 1);
    let e_exp_abs = if e_exp < 0 { -e_exp } else { e_exp };
    let sign_len = usize::from(sign);
    let decimal_point_len = usize::from(digit_len > 1);
    let exp_e_len = 1;
    let exp_sign_len = 1;
    let exp_sign_byte = if e_exp < 0 { b'-' } else { b'+' };
    let exp_digit_len = if e_exp_abs >= 1000 {
        4
    } else if e_exp_abs >= 100 {
        3
    } else if e_exp_abs >= 10 {
        2
    } else {
        1
    };
    let total_len =
        sign_len + decimal_point_len + digit_len + exp_e_len + exp_sign_len + exp_digit_len;
    out[0] = b'-';
    u128_to_ascii(digit_len, dw1, dw0, out, sign_len + decimal_point_len);
    if decimal_point_len > 0 {
        out[sign_len] = out[sign_len + 1];
        out[sign_len + 1] = b'.';
    }
    let i_e = sign_len + decimal_point_len + digit_len;
    out[i_e] = b'E';
    out[i_e + 1] = exp_sign_byte;
    render_tail_digits_before_index(exp_digit_len, e_exp_abs as u64, out, total_len);
    total_len
}

/// Finite non-zero: pick the form. Integer when the quantum exponent is zero, fixed-point when
/// it is negative and the adjusted exponent is at least -6, scientific otherwise -- the GDAS
/// auto-select rule.
fn fnz_ascii(out: &mut [u8], sign: bool, q_exp: i32, dw1: u64, dw0: u64) -> usize {
    let digit_len = calc_digit_len_128(dw1, dw0);
    let e_exp = q_exp + (digit_len as i32 - 1);
    if q_exp == 0 {
        return fnz_to_integer_ascii(out, sign, digit_len, dw1, dw0);
    }
    if q_exp <= 0 && e_exp >= -6 {
        return fnz_to_decimal_point_ascii(out, sign, q_exp, digit_len, dw1, dw0);
    }
    fnz_to_normalized_scientific_ascii(out, sign, q_exp, digit_len, dw1, dw0)
}

/* ------------------------------------------------------------------------
 * bits -> text
 * ---------------------------------------------------------------------- */

/// Render a BID128 encoding into the caller's buffer. Field extraction follows IEEE 754-2019
/// section 3.5.2; the class it lands on -- Infinity, NaN (quiet or signaling, with optional
/// payload), zero (canonical or non-canonical), or finite non-zero -- picks the leaf renderer.
fn decode_bid128_to_ascii(out: &mut [u8], bid128_hi: u64, bid128_lo: u64) -> usize {
    // The API's contract is a buffer of at least MAX_LEN, so that is what is
    // checked -- once, on every call, rather than per rendered form. The
    // public entry points take a `&mut [u8; MAX_LEN]`, but that length is
    // gone by the time the slice reaches here, so it is re-established at
    // this boundary.
    demand(out.len() >= MAX_LEN, "the buffer is at least MAX_LEN bytes");

    verify!(1 + W5 + T_WIDTH == 128);

    let sign = (bid128_hi & SIGN_BIT) != 0;
    let combination = ((bid128_hi >> (T_WIDTH - 64)) & 0x1FFFF) as u32;
    let coeff_t_hi = bid128_hi & ((1u64 << (T_WIDTH - 64)) - 1);

    let coeff_hi: u64;
    let q_exp: i32;

    if (combination >> (W5 - 2)) != 0x3 {
        // G0 G1 is 00, 01 or 10: the exponent is the top of the field and the
        // coefficient's three high bits are the bottom of it.
        let biased_exponent_e = (combination >> 3) as i32;
        q_exp = biased_exponent_e - BIAS;
        coeff_hi = (combination & 0x07) as u64;
    } else if (combination >> (W5 - 4)) != 0xF {
        // G0 G1 is 11 and G2 G3 is not: the coefficient's high bits are
        // 8 + Gw+4, and the exponent shifts down by two.
        let biased_exponent_e = ((combination >> 1) & ((1 << (W5 - 5 + 2)) - 1)) as i32;
        q_exp = biased_exponent_e - BIAS;
        coeff_hi = 8 + u64::from(combination & 1);
    } else if (combination >> (W5 - 5)) == 0x1E {
        return inf_ascii(out, sign);
    } else {
        verify!((combination >> (W5 - 5)) == 0x1F);
        let is_signaling = ((combination >> (W5 - 6)) & 1) != 0;
        return nan_ascii(out, sign, is_signaling, coeff_t_hi, bid128_lo);
    }

    let dw1 = (coeff_hi << (T_WIDTH - 64)) | coeff_t_hi;
    let dw0 = bid128_lo;

    // IEEE 754-2019 3.5.2: a coefficient at or above 10^34 is non-canonical and
    // the value used for it is zero, sign and exponent retained.
    if (dw1 | dw0) == 0
        || dw1 > MAXX_COEFF_34_HI
        || (dw1 == MAXX_COEFF_34_HI && dw0 >= MAXX_COEFF_34_LO)
    {
        return zer_ascii(out, sign, q_exp);
    }
    fnz_ascii(out, sign, q_exp, dw1, dw0)
}

/// Render a DPD128 encoding into the caller's buffer. The peer of `decode_bid128_to_ascii`:
/// both reduce an encoding to (sign, qExp, coefficient) and then dispatch to the same leaf
/// renderers, which have never seen an interchange format.
fn decode_dpd128_to_ascii(out: &mut [u8], dpd128_hi: u64, dpd128_lo: u64) -> usize {
    // The API's contract is a buffer of at least MAX_LEN, so that is what is
    // checked -- once, on every call, rather than per rendered form. The
    // public entry points take a `&mut [u8; MAX_LEN]`, but that length is
    // gone by the time the slice reaches here, so it is re-established at
    // this boundary.
    demand(out.len() >= MAX_LEN, "the buffer is at least MAX_LEN bytes");

    const W: u32 = 12; // exponent continuation width

    let sign = (dpd128_hi & SIGN_BIT) != 0;
    let combination = ((dpd128_hi >> (T_WIDTH - 64)) & 0x1FFFF) as u32;
    // in DPD the low W bits of the biased exponent always sit at the bottom of G
    let exp_continuation = (combination & 0x0FFF) as i32;

    let d0: u64;
    let q_exp: i32;
    let mut is_nan = false;
    let mut is_signaling = false;

    if (combination >> (W5 - 2)) != 0x3 {
        // leading digit 0..=7, exponent MSBs in G0 G1
        let exp_msb = (combination >> (W5 - 2)) as i32;
        q_exp = ((exp_msb << W) | exp_continuation) - BIAS;
        d0 = u64::from((combination >> W) & 0x7);
    } else if (combination >> (W5 - 4)) != 0xF {
        // leading digit 8 or 9, exponent MSBs displaced into G2 G3
        let exp_msb = ((combination >> (W5 - 4)) & 0x3) as i32;
        q_exp = ((exp_msb << W) | exp_continuation) - BIAS;
        d0 = 8 + u64::from((combination >> W) & 0x1);
    } else if (combination >> (W5 - 5)) == 0x1E {
        return inf_ascii(out, sign);
    } else {
        verify!((combination >> (W5 - 5)) == 0x1F);
        is_nan = true;
        is_signaling = ((combination >> (W5 - 6)) & 1) != 0;
        d0 = 0;
        q_exp = 0;
    }

    // Rebuild the binary coefficient from the declets. The leading digit rides
    // along as a sixth declet of the high group: 0..=9 encodes to itself.
    let declets_hi = (d0 << 50) | ((dpd128_hi & 0x3FFF_FFFF_FFFF) << 4) | (dpd128_lo >> 60);
    let declets_lo = dpd128_lo & 0x0FFF_FFFF_FFFF_FFFF; // low 60 bits, six declets
    let bin_hi = bin_from_declets(declets_hi);
    let bin_lo = bin_from_declets(declets_lo);

    let dw1: u64;
    let dw0: u64;
    if bin_hi == 0 {
        dw1 = 0;
        dw0 = bin_lo;
    } else {
        let prod = (bin_hi as u128) * (TEN_POW_18 as u128);
        let (sum_lo, carry) = (prod as u64).overflowing_add(bin_lo);
        dw1 = ((prod >> 64) as u64) + u64::from(carry);
        dw0 = sum_lo;
    }

    if is_nan {
        return nan_ascii(out, sign, is_signaling, dw1, dw0);
    }
    if (dw1 | dw0) == 0 {
        return zer_ascii(out, sign, q_exp);
    }
    fnz_ascii(out, sign, q_exp, dw1, dw0)
}

/* --------------------------------------------------------------------------
 * Packing a finalized value into an encoding
 *
 * Everything upstream of these -- parsing, rounding, clamping, the underflow
 * region -- is format-neutral; the two formats diverge only here.
 * ------------------------------------------------------------------------ */

/// Pack the sign, biased quantum exponent, and 113-bit coefficient into the BID128 encoding.
/// Always uses the short-form encoding (combination field top bits not 11), which is correct
/// for any canonical coefficient whose top 3 bits fit in 0..7. Preconditions: the exponent must
/// be in [-6176, 6111] and the coefficient must have at most 34 decimal digits.
fn bid128_set_finite(sign: bool, q_exp: i32, dw1: u64, dw0: u64) -> (u64, u64) {
    verify!((Q_MIN..=Q_MAX).contains(&q_exp));
    verify!(calc_digit_len_128(dw1, dw0) <= 34);
    let q_biased = (q_exp + BIAS) as u64;
    let s = if sign { SIGN_BIT } else { 0 };
    let g = q_biased << (3 + 46);
    (s | g | dw1, dw0)
}

/// Pack a finite value into DPD128. The peer of `bid128_set_finite`: same inputs, same sign
/// bit, same 14-bit biased exponent, and a coefficient of the same 34 digits -- laid out as a
/// leading digit carried in the combination field plus eleven declets. The leading digit cannot
/// be packed independently of the exponent: DPD puts it in the same five-bit prefix as the
/// exponent's top two bits, so unlike BID there is no shared "pack sign and exponent" step to
/// factor out.
#[cfg_attr(not(feature = "verify"), allow(unused_variables))]
fn dpd128_set_finite(sign: bool, q_exp: i32, dw1: u64, dw0: u64) -> (u64, u64) {
    // t_hi below is read only by a verify!, so with the feature off it is an unused binding.
    // Scoped here rather than crate-wide, and the function is short enough that nothing else
    // can hide behind it.
    verify!((Q_MIN..=Q_MAX).contains(&q_exp));
    verify!(calc_digit_len_128(dw1, dw0) <= 34);

    // Split the coefficient into its low 18 digits and its high 16. Two
    // divisions by 10^9 rather than one by 10^18, so the existing 128-bit kernel
    // does the work and no wider machinery is needed.
    let (t_hi, t_lo, r1) = u113_div_mod_pow10_max9(dw1, dw0, 9);
    let (t_hi, t_lo, r2) = u113_div_mod_pow10_max9(t_hi, t_lo, 9);
    let bin_lo = (r2 * 1_000_000_000) + r1;
    verify!(t_hi == 0);
    let bin_hi = t_lo;

    // the top digit rides in the combination field; the other 15 are declets
    let d0 = bin_hi / TEN_POW_15;
    let declets5_hi = declets_from_bin(bin_hi - (d0 * TEN_POW_15));
    let declets6_lo = declets_from_bin(bin_lo);
    verify!(d0 <= 9);

    let biased_q_exp = q_exp + BIAS;
    let exp_msb = (biased_q_exp >> 12) & 0x3;
    let exp_lsb = biased_q_exp & 0x0FFF;
    let g_prefix = if d0 <= 7 {
        (exp_msb << 3) | d0 as i32 // G0G1 = exponent MSBs
    } else {
        0x18 | (exp_msb << 1) | (d0 as i32 & 1) // G0G1 = 11, digit is 8 or 9
    };
    let sign_value = u64::from(sign);
    let sign_combo = (sign_value << 17) | ((g_prefix as u64) << 12) | (exp_lsb as u64);

    (
        (sign_combo << 46) | (declets5_hi >> 4),
        (declets5_hi << 60) | declets6_lo,
    )
}

/// Pack a finite value into whichever interchange format the caller asked for. BID and DPD are
/// peers: both carry the same sign, the same 14-bit biased exponent and the same 34-digit
/// coefficient, and differ only in how that coefficient is laid out. Everything upstream of
/// this point -- rounding, clamping and the underflow region -- is format-neutral.
fn set_finite(format_is_bid: bool, sign: bool, q_exp: i32, dw1: u64, dw0: u64) -> (u64, u64) {
    if format_is_bid {
        bid128_set_finite(sign, q_exp, dw1, dw0)
    } else {
        dpd128_set_finite(sign, q_exp, dw1, dw0)
    }
}

fn bid128_set_nan(
    sign: bool,
    is_signaling: bool,
    payload_dw1: u64,
    payload_dw0: u64,
) -> (u64, u64) {
    let without_payload = (if is_signaling {
        SNAN_PATTERN_HI
    } else {
        QNAN_PATTERN_HI
    }) | (if sign { SIGN_BIT } else { 0 });
    (without_payload | payload_dw1, payload_dw0)
}

/// Pack a NaN into DPD128. The payload occupies the whole 110-bit trailing field as eleven
/// declets -- 33 digits, with no leading digit in the combination field, since G0..G4 are fixed
/// at 11111 and G5 carries the signaling flag.
#[cfg_attr(not(feature = "verify"), allow(unused_variables))]
fn dpd128_set_nan(
    sign: bool,
    is_signaling: bool,
    payload_dw1: u64,
    payload_dw0: u64,
) -> (u64, u64) {
    // Same verify!-only binding as dpd128_set_finite; see the note there.
    verify!(calc_digit_len_128(payload_dw1, payload_dw0) <= 33);

    let (t_hi, t_lo, r1) = u113_div_mod_pow10_max9(payload_dw1, payload_dw0, 9);
    let (t_hi, t_lo, r2) = u113_div_mod_pow10_max9(t_hi, t_lo, 9);
    let bin_lo = (r2 * 1_000_000_000) + r1;
    verify!(t_hi == 0);
    let declets5_hi = declets_from_bin(t_lo);
    let declets6_lo = declets_from_bin(bin_lo);

    let g_field = (0x1Fu64 << 12) | (if is_signaling { 1 << 11 } else { 0 });
    let sign_combo = (u64::from(sign) << 17) | g_field;
    (
        (sign_combo << 46) | (declets5_hi >> 4),
        (declets5_hi << 60) | declets6_lo,
    )
}

/// Pack a NaN into whichever interchange format the caller asked for. A NaN payload is a
/// coefficient, so the formats diverge here exactly as they do for finite values: BID keeps it
/// as a binary integer, DPD as declets. Infinity by contrast has no coefficient, which is why
/// it is byte-identical in both and needs no such split.
fn set_nan(
    format_is_bid: bool,
    sign: bool,
    is_signaling: bool,
    payload_dw1: u64,
    payload_dw0: u64,
) -> (u64, u64) {
    if format_is_bid {
        bid128_set_nan(sign, is_signaling, payload_dw1, payload_dw0)
    } else {
        dpd128_set_nan(sign, is_signaling, payload_dw1, payload_dw0)
    }
}

/* ------------------------------------------------------------------------
 * Finalize: rounding, clamping, underflow
 * ---------------------------------------------------------------------- */

/// Encode Infinity (or -Infinity).
fn finalize_overflow(sign: bool) -> (u64, u64) {
    let s = if sign { SIGN_BIT } else { 0 };
    (s | INF_PATTERN_HI, 0)
}

/// Test whether a coefficient/exponent pair fits as a canonical encoding: exponent in [-6176,
/// 6111] and coefficient in [0, 10^34). Fast path uses the leading-zero bit count of the high
/// word to cheaply confirm the coefficient has at most 113 bits, in which case it is
/// necessarily less than 10^34 (since 2^113 > 10^34). Slow path compares against 10^34.
fn coeff_q_exp_fit(dw1: u64, dw0: u64, q_exp: i32) -> bool {
    if !(Q_MIN..=Q_MAX).contains(&q_exp) {
        return false;
    }
    // at most 113 bits is necessarily below 10^34, since 2^113 > 10^34
    if dw1.leading_zeros() > 128 - 113 {
        return true;
    }
    dw1 < MAXX_COEFF_34_HI || (dw1 == MAXX_COEFF_34_HI && dw0 < MAXX_COEFF_34_LO)
}

/// Encode a finite value whose natural quantum exponent exceeds qMax (6111) but whose
/// coefficient has enough leading capacity to absorb the excess. Multiplies the coefficient by
/// 10^qExcess (where qExcess is qExp minus 6111), reducing the exponent to exactly 6111. This
/// preserves the value while bringing the encoding into representable range. Caller must have
/// verified that the coefficient has enough room (digitLen + qExcess <= 34); this function
/// asserts the invariant and produces the clamped encoding.
fn finalize_clamping(
    format_is_bid: bool,
    sign: bool,
    q_exp: i32,
    dw1: u64,
    dw0: u64,
) -> (u64, u64) {
    let q_excess = q_exp - Q_MAX;
    verify!(q_excess > 0 && q_excess as usize <= 34 - calc_digit_len_128(dw1, dw0));
    let pow10_offset = (q_excess as usize) << 1;
    let y1 = POW10[pow10_offset + 1];
    let y0 = POW10[pow10_offset];
    // The three cross terms all discard their high halves and the sum wraps:
    // that is relied on here, and the result is guaranteed in range by the
    // caller's digit-count test.
    let pp00 = (dw0 as u128) * (y0 as u128);
    let pp00_hi = (pp00 >> 64) as u64;
    let pp00_lo = pp00 as u64;
    let pp10_lo = dw1.wrapping_mul(y0);
    let pp01_lo = dw0.wrapping_mul(y1);
    let dw0_scaled = pp00_lo;
    let dw1_scaled = pp00_hi.wrapping_add(pp10_lo).wrapping_add(pp01_lo);
    verify!(coeff_q_exp_fit(dw1_scaled, dw0_scaled, Q_MAX));
    set_finite(format_is_bid, sign, Q_MAX, dw1_scaled, dw0_scaled)
}

/// Handle the underflow boundary case where the most-significant digit of the coefficient sits
/// exactly one position to the right of the representable range. The coefficient value itself
/// determines rounding: comparing the coefficient against (10^digitLen) / 2 yields the residue
/// class. Combined with any incoming residue, this drives a roundTiesToEven decision between
/// zero (always even, the default outcome) and 1 ulp (the smallest subnormal, only when total
/// residue is `GT_HALF`).
fn finalize_underflow_boundary(
    format_is_bid: bool,
    sign: bool,
    dw1: u64,
    dw0: u64,
    residue_in: i32,
) -> (u64, u64) {
    let digit_len = calc_digit_len_128(dw1, dw0);
    verify!(digit_len != 0);
    let scale_residue = residue_from_value_pow10(dw1, dw0, digit_len);
    let total_residue = residue_merge(scale_residue, residue_in);
    let coeff = u64::from(total_residue == GT_HALF);
    set_finite(format_is_bid, sign, Q_MIN, 0, coeff)
}

/// Representable as a subnormal: truncate to the minimum exponent, keeping what precision
/// survives, and round what is left.
fn finalize_subnormal(
    format_is_bid: bool,
    sign: bool,
    q_exp_in: i32,
    dw1_in: u64,
    dw0_in: u64,
    residue_in: i32,
) -> (u64, u64) {
    let truncation_needed = (Q_MIN - q_exp_in) as usize;
    verify!(truncation_needed > 0 && truncation_needed < calc_digit_len_128(dw1_in, dw0_in));

    let (mut dw1_t, mut dw0_t, scale_residue) =
        u128_scale_down_pow10(dw1_in, dw0_in, truncation_needed);
    let total_residue = residue_merge(scale_residue, residue_in);
    let mut q_exp_t = Q_MIN;

    let round_up = (total_residue == GT_HALF) || (total_residue == TIE_HALF && (dw0_t & 1) != 0);
    if round_up {
        // The increment carries out of the low word only when it is all ones,
        // which is let wrap to zero and caught with the test below.
        dw0_t = dw0_t.wrapping_add(1);
        if dw0_t == 0 {
            dw1_t += 1;
        }
        if dw1_t == MAXX_COEFF_34_HI && dw0_t == MAXX_COEFF_34_LO {
            dw1_t = MIN_PRECISION_34_COEFF_HI;
            dw0_t = MIN_PRECISION_34_COEFF_LO;
            q_exp_t += 1;
        }
    }
    set_finite(format_is_bid, sign, q_exp_t, dw1_t, dw0_t)
}

/// Dispatch underflow handling for a value whose exponent is below Emin (-6176). Three cases:
///
/// - Truncation needed exceeds the digit count: result is zero (the value is below half the
///   smallest subnormal).
/// - Truncation equals the digit count: boundary case where the leading digit of the
///   coefficient determines rounding.
/// - Truncation is less than the digit count: representable as a subnormal; some precision is
///   preserved, with rounding.
fn finalize_underflow_region(
    format_is_bid: bool,
    sign: bool,
    q_exp: i32,
    dw1: u64,
    dw0: u64,
    residue: i32,
) -> (u64, u64) {
    // IEEE 754-2019 7.5 Underflow -- the subnormal region
    verify!(q_exp < Q_MIN);
    let truncation_needed = (Q_MIN - q_exp) as usize;
    let digit_len = calc_digit_len_128(dw1, dw0);
    verify!(digit_len <= 34);
    if truncation_needed > digit_len {
        // below half the smallest subnormal: zero, and always inexact
        round_and_finalize_finite(format_is_bid, sign, Q_MIN, 0, 0, LT_HALF)
    } else if truncation_needed == digit_len {
        finalize_underflow_boundary(format_is_bid, sign, dw1, dw0, residue)
    } else {
        finalize_subnormal(format_is_bid, sign, q_exp, dw1, dw0, residue)
    }
}

/// Round and finalize a finite parsed coefficient/exponent pair into its encoding. Implements
/// roundTiesToEven rounding with full handling of the decimal128 boundary conditions: oversized
/// coefficient, underflow to subnormal or zero, overflow to infinity, and the clamped-near-Emax
/// region where a finite value with too-large quantum exponent can be representable by
/// absorbing the excess into the coefficient.
///
/// Step 1 (fast path): if the input already fits within precision and range and no rounding is
/// required (residue <= `LT_HALF`, or `HALF` with an even retained low bit), encode and return
/// immediately. This picks up the vast majority of cases.
///
/// Step 2 (zero coefficient): zeros that are out of qExp range get clamped to the valid range.
///
/// Step 3 (special values): not applicable here -- special values are handled by callers before
/// reaching this function.
///
/// Step 4 (underflow): if the exponent is below Emin, divert to `finalize_underflow_region` for
/// subnormal handling.
///
/// Step 5 (precision truncation): REMOVED as unreachable from a codec's callers -- see the note
/// in the body, which also states what would have to change for it to be needed again.
///
/// Step 6 (rounding): apply roundTiesToEven. If rounding causes the coefficient to roll over to
/// 10^34, normalize to 10^33 and increment the exponent.
///
/// Step 7 (overflow / clamping): if the exponent exceeds Emax, either absorb the excess into
/// the coefficient (clamping near Emax) or overflow to infinity.
fn round_and_finalize_finite(
    format_is_bid: bool,
    sign: bool,
    q_exp_in: i32,
    dw1_in: u64,
    dw0_in: u64,
    residue_in: i32,
) -> (u64, u64) {
    // Step 1
    if residue_in <= LT_HALF || (residue_in == TIE_HALF && (dw0_in & 1) == 0) {
        if coeff_q_exp_fit(dw1_in, dw0_in, q_exp_in) {
            return set_finite(format_is_bid, sign, q_exp_in, dw1_in, dw0_in);
        }
        // Step 2
        if (dw1_in | dw0_in) == 0 {
            let q_clamped = q_exp_in.clamp(Q_MIN, Q_MAX);
            return set_finite(format_is_bid, sign, q_clamped, 0, 0);
        }
    }

    /*
     * Step 5 -- precision truncation -- IS NOT HERE, and this is the reason.
     *
     * The finalizer this descends from serves an arithmetic engine, whose
     * callers hand it coefficients wider than the 34-digit precision limit:
     * multiply and FMA produce a double-width product, and the engine truncates
     * it here, accumulating the discarded digits into the residue before
     * rounding. A codec has no such caller.
     *
     * The parser accumulates at most 19 + 15 = 34 significant digits and turns
     * every digit past them into a residue instead, so the coefficient it hands
     * over is never wider than the limit. The only other entry to this function
     * is the swamped-underflow branch above, which passes a zero coefficient. So
     * the truncation step is unreachable, and rather than ship a path that has
     * never once executed -- and therefore has never been verified to work -- it
     * is removed and the invariant is enforced.
     *
     * Measured: zero occurrences across the 87k-row corpus and
     * 400,000 random numerals, and across 76,280 finalizer calls driven by
     * inputs of up to 5,000 digits, where the widest coefficient ever seen was
     * 34.
     *
     * This is `demand`, not `verify!`: it must survive into shipping builds,
     * because falling through would encode an out-of-range coefficient
     * silently. It cannot fire on a consumer's data -- no input reaches it --
     * only on a future caller that breaks the invariant, which is exactly when a
     * loud failure is what you want. A caller that is not a parser, or one that
     * parses differently, needs the truncation step back.
     */
    demand(
        calc_digit_len_128(dw1_in, dw0_in) <= 34,
        "the coefficient is at most 34 digits",
    );

    // Step 4 -- with no precision truncation to outrank it, the underflow test
    // is simply whether the exponent is below the minimum.
    if Q_MIN - q_exp_in > 0 {
        return finalize_underflow_region(
            format_is_bid,
            sign,
            q_exp_in,
            dw1_in,
            dw0_in,
            residue_in,
        );
    }

    let total_residue = residue_in;
    let mut dw1 = dw1_in;
    let mut dw0 = dw0_in;
    let mut q_exp = q_exp_in;

    // Step 6 -- ties-to-even, the only rounding this codec has
    let apply_rounding =
        (total_residue == GT_HALF) || (total_residue == TIE_HALF && (dw0 & 1) != 0);
    if apply_rounding {
        // As in finalize_subnormal: the low word wraps to zero exactly when it
        // was all ones, and the carry is taken from that.
        dw0 = dw0.wrapping_add(1);
        if dw0 == 0 {
            dw1 += 1;
        }
        if dw1 == MAXX_COEFF_34_HI && dw0 == MAXX_COEFF_34_LO {
            dw1 = MIN_PRECISION_34_COEFF_HI;
            dw0 = MIN_PRECISION_34_COEFF_LO;
            q_exp += 1;
        }
    }

    // Step 7
    verify!(q_exp >= Q_MIN);
    verify!(calc_digit_len_128(dw1, dw0) <= 34);
    if q_exp > Q_MAX {
        let q_excess = (q_exp - Q_MAX) as usize;
        let digit_len = calc_digit_len_128(dw1, dw0);
        if digit_len + q_excess <= 34 {
            return finalize_clamping(format_is_bid, sign, q_exp, dw1, dw0);
        }
        return finalize_overflow(sign);
    }
    set_finite(format_is_bid, sign, q_exp, dw1, dw0)
}

/* ------------------------------------------------------------------------
 * text -> bits
 * ---------------------------------------------------------------------- */

/// The code of the character at `index`, or -1 past the end of the input. The parsers walk the
/// input as character codes, so end-of-input is one more code value rather than a bound test at
/// every read.
fn code_at(str: &[u8], index: usize) -> i32 {
    // -1 is not a byte, so a NUL in the string stays an ordinary -- and invalid -- input byte
    // rather than doubling as the terminator: a &str carries its own length and can hold one,
    // and it is rejected. Parsing walks the UTF-8 bytes rather than the chars, on a borrowed
    // slice, so nothing is copied. Rust has no increment operator, so the advance is its own
    // line at each call rather than riding in the index.
    if index < str.len() {
        i32::from(str[index])
    } else {
        -1
    }
}

/// The digit test, spelled out once: `true` for '0' through '9' and nothing else.
fn is_digit(ch: i32) -> bool {
    // u8::is_ascii_digit has no counterpart on the i32 the reader yields.
    ch >= i32::from(b'0') && ch <= i32::from(b'9')
}

/// Parse a finite decimal numeral and produce the corresponding encoding. Accepts an optional
/// decimal point, an optional E/e exponent (with optional +/- sign), and any number of
/// significant digits. Coefficients with more than 34 significant digits are accepted; leading
/// zeros before a non-zero digit are flushed; the digits past 34 are properly tracked as
/// residue for subsequent roundTiesToEven rounding. Excess exponent magnitudes are clamped to
/// 9999, which is sufficient for the rounding logic to detect overflow or underflow.
///
/// Sign is taken from the `sign` parameter (already consumed by the dispatcher); the leading
/// character `ch_first` is the first digit or `.` and is processed by the main loop.
fn parse_finite_value_text(
    format_is_bid: bool,
    sign: bool,
    ch_first: i32,
    bytes: &[u8],
    start_pos: usize,
) -> Result<(u64, u64), &'static str> {
    // Leading zeros can be flushed branchlessly by coercing a comparison to an integer; the two
    // counters below say the same thing with an if, since usize::from(bool) in an accumulator
    // reads worse than the branch.
    let mut pos = start_pos;
    let mut residue = EXACT;

    let mut has_coefficient_digit = false; // any digit at all, including zero
    let mut significant_digit_count: usize = 0; // does not count leading zeros
    let mut has_dot = false;
    let mut exp_sign = false;
    // an underscore is a digit separator: legal only between two digits, so it
    // must follow a digit and the next character must be a digit too
    let mut prev_was_digit = false;
    let mut after_underscore = false;

    verify!(ch_first != -1);
    let mut ch = ch_first;
    let mut fractional_digit_count: usize = 0;
    let mut accum19a: u64 = 0;
    let mut accum19b: u64 = 0;
    let mut exp: i32 = 0;

    while is_digit(ch) || ch == i32::from(b'.') || ch == i32::from(b'_') {
        if is_digit(ch) {
            let d = (ch - i32::from(b'0')) as u32;
            has_coefficient_digit = true;
            // count while flushing leading zeros
            if significant_digit_count != 0 || d != 0 {
                significant_digit_count += 1;
            }
            if significant_digit_count <= 19 {
                accum19a = (accum19a * 10) + u64::from(d);
            } else if significant_digit_count <= 34 {
                accum19b = (accum19b * 10) + u64::from(d);
            } else if significant_digit_count == 34 + 1 {
                residue = residue_from_decimal_digit(d);
            } else {
                residue = residue_merge(residue, residue_from_decimal_digit(d));
            }
            if has_dot {
                fractional_digit_count += 1;
            }
            prev_was_digit = true;
            after_underscore = false;
        } else if ch == i32::from(b'_') {
            if !prev_was_digit {
                return Err(ERR_UNDERSCORE_LOCATION);
            }
            prev_was_digit = false;
            after_underscore = true;
        } else {
            // '.'
            if after_underscore {
                return Err(ERR_UNDERSCORE_LOCATION);
            }
            if has_dot {
                return Err(ERR_DOUBLE_RADIX_DOT);
            }
            has_dot = true;
            prev_was_digit = false;
        }
        ch = code_at(bytes, pos);
        pos += 1;
    }
    if after_underscore {
        return Err(ERR_UNDERSCORE_LOCATION);
    }
    if !has_coefficient_digit {
        return Err(ERR_NO_COEFFICIENT_DIGIT);
    }
    // this path has at least one digit
    if ch == i32::from(b'E') || ch == i32::from(b'e') {
        ch = code_at(bytes, pos);
        pos += 1;
        if ch == i32::from(b'+') || ch == i32::from(b'-') {
            exp_sign = ch == i32::from(b'-');
            ch = code_at(bytes, pos);
            pos += 1;
        }
        let mut has_exp_digit = false;
        let mut exp_significant_digit_count: usize = 0;
        prev_was_digit = false;
        after_underscore = false;
        while is_digit(ch) || ch == i32::from(b'_') {
            if ch == i32::from(b'_') {
                if !prev_was_digit {
                    return Err(ERR_UNDERSCORE_LOCATION);
                }
                prev_was_digit = false;
                after_underscore = true;
            } else {
                has_exp_digit = true;
                let e_digit = ch - i32::from(b'0');
                // count while flushing leading zeros
                if exp_significant_digit_count != 0 || e_digit != 0 {
                    exp_significant_digit_count += 1;
                }
                // A long exponent field runs this past the width of an i32; the
                // reference wraps here and so does this, because the value is
                // thrown away by the clamp below the moment the field is more
                // than four significant digits wide.
                exp = exp.wrapping_mul(10).wrapping_add(e_digit);
                prev_was_digit = true;
                after_underscore = false;
            }
            ch = code_at(bytes, pos);
            pos += 1;
        }
        if after_underscore {
            return Err(ERR_UNDERSCORE_LOCATION);
        }
        if !has_exp_digit {
            return Err(ERR_NO_EXPONENT_DIGIT);
        }
        // clamp once after the loop
        if exp_significant_digit_count > 4 {
            exp = 9999;
        }
    }
    if ch != -1 {
        return Err(ERR_UNEXPECTED_CHAR);
    }

    let mut dw0 = accum19a;
    let mut dw1: u64 = 0;
    if significant_digit_count > 19 {
        let capped = significant_digit_count.min(34);
        (dw1, dw0) = u128_fma_pow10(accum19a, capped - 19, accum19b);
    }

    /*
     * Excess digits to the right of the radix point and excess digits beyond the
     * precision limit pull the quantum exponent in opposite directions, so both
     * corrections land here.
     */
    let signed_exp = if exp_sign { -exp } else { exp };
    let excess_digits = significant_digit_count.max(34) - 34;
    let q_exp = signed_exp - fractional_digit_count as i32 + excess_digits as i32;
    Ok(round_and_finalize_finite(
        format_is_bid,
        sign,
        q_exp,
        dw1,
        dw0,
        residue,
    ))
}

/// Parse the keyword Infinity (or its 3-letter abbreviation Inf), case-insensitively, and
/// produce the corresponding encoding. Any other input is rejected.
fn parse_infinity_text(
    sign: bool,
    ch_first: i32,
    bytes: &[u8],
    start_pos: usize,
) -> Result<(u64, u64), &'static str> {
    let mut ch = ch_first;
    let mut ch_prev_code = 0i32;
    const TARGET: &[u8; 8] = b"infinity";
    let mut pos = start_pos;
    let mut i = 0;
    while i < 8 {
        let target_code = TARGET[i];
        if (ch | 0x20) != i32::from(target_code) {
            if ch == -1 {
                break;
            }
            return Err(ERR_PARSING_INFINITY);
        }
        ch_prev_code = ch | 0x20;
        ch = code_at(bytes, pos);
        pos += 1;
        i += 1;
    }
    if ch != -1 || (ch_prev_code != i32::from(b'f') && ch_prev_code != i32::from(b'y')) {
        return Err(ERR_PARSING_INFINITY);
    }
    let s = if sign { SIGN_BIT } else { 0 };
    Ok((INF_PATTERN_HI | s, 0))
}

/// Parse NaN, qNaN, or sNaN (case-insensitive) with an optional decimal payload, and produce
/// the corresponding encoding. Payloads of 34+ digits AS WRITTEN are non-canonical and produce
/// a zero payload per IEEE 754-2019 section 3.5.2.
///
/// The payload may be written bare (`NaN123`) or enclosed in parentheses (`NaN(123)`). The
/// parenthesised form is positional and balanced: the `(` must follow the word immediately, at
/// least one digit must appear, and the `)` must be the last character of the input. So
/// `NaN(1)2(3)`, `NaN(1`, `NaN(1]` and `NaN()` are all rejected, and `NaN[1]` / `NaN{1}` are
/// not this form at all.
///
/// The reason this spelling exists at all is that the platform's own `strtod` reads
/// `NAN(n-char-sequence)`, so the spelling reaches a caller from the platform itself. The
/// resemblance stops at the parentheses -- `strtod`'s sequence is letters, digits and
/// underscores read as an implementation-defined integer (`NAN(abc)` and `NAN(0x1c8)` are legal
/// there), while a decimal128 payload is decimal digits. This is a paren-tolerant spelling of
/// our own payload syntax, not `strtod`'s.
fn parse_nan_text(
    format_is_bid: bool,
    sign: bool,
    ch_first: i32,
    bytes: &[u8],
    start_pos: usize,
) -> Result<(u64, u64), &'static str> {
    let mut pos = start_pos;
    let mut ch = ch_first;
    let has_s = (ch | 0x20) == i32::from(b's');
    let has_q = (ch | 0x20) == i32::from(b'q');
    if has_q || has_s {
        ch = code_at(bytes, pos);
        pos += 1;
    }
    if ((ch | 0x20) != i32::from(b'n'))
        || ((code_at(bytes, pos) | 0x20) != i32::from(b'a'))
        || ((code_at(bytes, pos + 1) | 0x20) != i32::from(b'n'))
    {
        return Err(ERR_PARSING_NAN);
    }
    pos += 2;

    let mut payload_dw0: u64 = 0;
    let mut payload_dw1: u64 = 0;
    ch = code_at(bytes, pos);
    pos += 1;
    let has_paren = ch == i32::from(b'(');
    if has_paren {
        ch = code_at(bytes, pos);
        pos += 1;
    }
    if has_paren || ch != -1 {
        let mut accum_digit_count: usize = 0;
        let mut accum19a: u64 = 0;
        let mut accum19b: u64 = 0;
        while is_digit(ch) {
            let d = (ch - i32::from(b'0')) as u64;
            // Count every digit written, leading zeros included. The 33-digit
            // limit is a limit on the payload as spelled, not on its significant
            // digits, so "NaN" followed by forty zeros and a one is over the
            // limit and reads as zero. Leading zeros contribute nothing to the
            // accumulators, so the value of a payload within the limit is
            // unaffected -- and an over-limit payload runs the second
            // accumulator past 64 bits, which is let wrap and which the limit
            // test below discards.
            accum_digit_count += 1;
            if accum_digit_count <= 19 {
                accum19a = (accum19a * 10) + d;
            } else {
                accum19b = accum19b.wrapping_mul(10).wrapping_add(d);
            }
            ch = code_at(bytes, pos);
            pos += 1;
        }
        if accum_digit_count == 0 {
            return Err(ERR_PARSING_NAN);
        }
        if has_paren {
            // the ')' must be present and must end the input
            if ch != i32::from(b')') {
                return Err(ERR_PARSING_NAN);
            }
            if code_at(bytes, pos) != -1 {
                return Err(ERR_PARSING_NAN);
            }
        } else if ch != -1 {
            return Err(ERR_PARSING_NAN);
        }
        if accum_digit_count > 33 {
            payload_dw0 = 0;
            payload_dw1 = 0;
        } else {
            payload_dw0 = accum19a;
            payload_dw1 = 0;
            if accum_digit_count > 19 {
                (payload_dw1, payload_dw0) =
                    u128_fma_pow10(accum19a, accum_digit_count - 19, accum19b);
            }
        }
    }
    Ok(set_nan(
        format_is_bid,
        sign,
        has_s,
        payload_dw1,
        payload_dw0,
    ))
}

/// Top-level dispatcher for parsing a decimal character sequence into an encoding. Reads
/// optional sign, then dispatches by leading character to the finite-value, infinity, or NaN
/// parser. The leading character is also passed through to the dispatched parser so it doesn't
/// have to re-read what the dispatcher already consumed.
fn encode_string_or_error(format_is_bid: bool, bytes: &[u8]) -> Result<(u64, u64), &'static str> {
    let mut pos = 0;
    let ch_first = code_at(bytes, pos);
    pos += 1;
    let mut ch = ch_first;
    let sign = ch == i32::from(b'-');
    if ch == i32::from(b'+') || ch == i32::from(b'-') {
        ch = code_at(bytes, pos);
        pos += 1;
    }
    if is_digit(ch) || ch == i32::from(b'.') {
        return parse_finite_value_text(format_is_bid, sign, ch, bytes, pos);
    }
    let ch_lower = ch | 0x20;
    if ch_lower == i32::from(b'i') {
        return parse_infinity_text(sign, ch, bytes, pos);
    }
    if ch_lower == i32::from(b'n') || ch_lower == i32::from(b's') || ch_lower == i32::from(b'q') {
        return parse_nan_text(format_is_bid, sign, ch, bytes, pos);
    }
    if ch_first == -1 {
        return Err(ERR_EMPTY_STRING);
    }
    Err(ERR_INVALID_FORMAT)
}
