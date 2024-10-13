using System;

namespace Doji.AI.Safetensors {

    /// <summary>
    /// Represents the various data types available in safetensors.
    /// </summary>
    public enum Dtype {
        /// <summary>
        /// Boolean type
        /// </summary>
        BOOL,

        /// <summary>
        /// Unsigned byte
        /// </summary>
        U8,

        /// <summary>
        /// Signed byte
        /// </summary>
        I8,

        /// <summary>
        /// FP8 (E5M2) Floating point based on paper: https://arxiv.org/pdf/2209.05433.pdf
        /// </summary>
        F8_E5M2,

        /// <summary>
        /// FP8 (E4M3) Floating point based on paper: https://arxiv.org/pdf/2209.05433.pdf
        /// </summary>
        F8_E4M3,

        /// <summary>
        /// Signed integer (16-bit)
        /// </summary>
        I16,

        /// <summary>
        /// Unsigned integer (16-bit)
        /// </summary>
        U16,

        /// <summary>
        /// Half-precision floating point
        /// </summary>
        F16,

        /// <summary>
        /// Brain floating point
        /// </summary>
        BF16,

        /// <summary>
        /// Signed integer (32-bit)
        /// </summary>
        I32,

        /// <summary>
        /// Unsigned integer (32-bit)
        /// </summary>
        U32,

        /// <summary>
        /// Floating point (32-bit)
        /// </summary>
        F32,

        /// <summary>
        /// Floating point (64-bit)
        /// </summary>
        F64,

        /// <summary>
        /// Signed integer (64-bit)
        /// </summary>
        I64,

        /// <summary>
        /// Unsigned integer (64-bit)
        /// </summary>
        U64
    }

    /// <summary>
    /// Provides the size in bytes for each Dtype.
    /// </summary>
    public static class DtypeExtensions {
        public static int Size(this Dtype dtype) {
            return dtype switch {
                Dtype.BOOL => 1,
                Dtype.U8 => 1,
                Dtype.I8 => 1,
                Dtype.F8_E5M2 => 1,
                Dtype.F8_E4M3 => 1,
                Dtype.I16 => 2,
                Dtype.U16 => 2,
                Dtype.F16 => 2,
                Dtype.BF16 => 2,
                Dtype.I32 => 4,
                Dtype.U32 => 4,
                Dtype.F32 => 4,
                Dtype.F64 => 8,
                Dtype.I64 => 8,
                Dtype.U64 => 8,
                _ => throw new NotSupportedException($"Unsupported Dtype: {dtype}")
            };
        }
    }
}