using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace Doji.AI.Safetensors {

    /// <summary>
    /// Represents errors that occur during safetensors operations.
    /// </summary>
    public class SafeTensorException : Exception {
        public SafeTensorException() { }

        public SafeTensorException(string message) : base(message) { }

        public SafeTensorException(string message, Exception inner) : base(message, inner) { }
    }

    public class InvalidHeaderException : SafeTensorException {
        public InvalidHeaderException() : base("The header is an invalid UTF-8 string and cannot be read.") { }
        public InvalidHeaderException(Exception inner) : base("The header is an invalid UTF-8 string and cannot be read.", inner) { }
    }

    public class InvalidHeaderStartException : SafeTensorException {
        public InvalidHeaderStartException() : base("The header's first byte is not the expected `{`.") { }
    }

    public class InvalidHeaderDeserializationException : SafeTensorException {
        public InvalidHeaderDeserializationException() : base("The header does contain a valid string, but it is not valid JSON.") { }
    }

    public class HeaderTooLargeException : SafeTensorException {
        public HeaderTooLargeException() : base("The header is larger than 100MB, which is considered too large.") { }
    }

    public class HeaderTooSmallException : SafeTensorException {
        public HeaderTooSmallException() : base("The header is smaller than 8 bytes.") { }
    }

    public class InvalidHeaderLengthException : SafeTensorException {
        public InvalidHeaderLengthException() : base("The header length is invalid.") { }
    }

    public class TensorNotFoundException : SafeTensorException {
        public TensorNotFoundException(string tensorName)
            : base($"The tensor '{tensorName}' was not found in the archive.") { }
    }

    public class TensorInvalidInfoException : SafeTensorException {
        public TensorInvalidInfoException() : base("Invalid information between shape, dtype, and the proposed offsets in the file.") { }
    }

    public class InvalidOffsetException : SafeTensorException {
        public InvalidOffsetException(string tensorName)
            : base($"The offsets declared for the tensor '{tensorName}' in the header are invalid.") { }
    }

    public class IoErrorException : SafeTensorException {
        public IoErrorException(IOException innerException)
            : base("An I/O error occurred.", innerException) { }
    }

    public class JsonErrorException : SafeTensorException {
        public JsonErrorException(JsonException innerException)
            : base("A JSON error occurred.", innerException) { }
    }

    public class InvalidTensorViewException : SafeTensorException {
        public InvalidTensorViewException(string dtype, List<int> shape, int bufferSize)
            : base($"The tensor with dtype '{dtype}', shape {string.Join(",", shape)}, and buffer size {bufferSize} is invalid.") { }
    }

    public class MetadataIncompleteBufferException : SafeTensorException {
        public MetadataIncompleteBufferException()
            : base("The metadata is invalid because the data offsets of the tensor do not fully cover the buffer part of the file.") { }
    }

    public class ValidationOverflowException : SafeTensorException {
        public ValidationOverflowException()
            : base("The metadata contains information (shape or shape * dtype size) that leads to an arithmetic overflow.") { }
    }
}