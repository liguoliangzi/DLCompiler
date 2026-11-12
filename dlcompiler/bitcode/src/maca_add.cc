// ============================================================================
// MACA (MetaX) custom op bitcode library for dl.custom().
//
// The MACA pipeline lowers dl.custom() to per-element extern calls of the
// user symbol, so the bitcode library exports plain scalar functions named
// after the registered symbol:
//
//   Python:  self.symbol = "custom_add_" + str(a.dtype)
//            str(tl.float32) -> "fp32"   -> custom_add_fp32
//            str(tl.int32)   -> "int32"  -> custom_add_int32
//            str(tl.float16) -> "fp16"   -> custom_add_fp16
//
// Build: bash dlcompiler/bitcode/compile_maca_bc.sh
// Output: language/deeplink/bitcode/bc/maca_add.bc (auto-resolved by name)
// ============================================================================

extern "C" {

float custom_add_fp32(float a, float b) { return a + b; }

int custom_add_int32(int a, int b) { return a + b; }

__fp16 custom_add_fp16(__fp16 a, __fp16 b) {
  return static_cast<__fp16>(static_cast<float>(a) + static_cast<float>(b));
}

} // extern "C"