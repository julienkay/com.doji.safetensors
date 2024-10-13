using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Doji.AI.Safetensors {

    /// <summary>
    /// Contains information about a single tensor.
    /// </summary>
    public class TensorInfo {
        [JsonProperty("dtype")]
        public Dtype Dtype { get; set; }

        [JsonProperty("shape")]
        public ulong[] Shape { get; set; }

        [JsonProperty("data_offsets")]
        public ulong[] DataOffsets { get; set; }
    }

    /// <summary>
    /// The stuct representing the header of safetensor files which allow
    /// indexing into the raw byte-buffer array and how to interpret it.
    /// </summary>
    public class Metadata {
        public Dictionary<string, string> MetadataInfo { get; set; }
        public Dictionary<string, TensorInfo> Tensors { get; set; }
        public Dictionary<string, uint> IndexMap { get; set; }

        public Metadata(Dictionary<string, TensorInfo> tensors) {
            IndexMap = new Dictionary<string, uint>(tensors.Count);
            Tensors = tensors;

            uint i = 0;
            foreach (var info in tensors) {
                IndexMap.Add(info.Key, i++);
            }
        }
    }

    /// <summary>
    /// Represents a readable view of a tensor's data within the file.
    /// </summary>
    public class TensorView {
        public Dtype Dtype { get; }
        public ulong[] Shape { get; }
        public ReadOnlyMemory<byte> Data { get; }

        public TensorView(Dtype dtype, ulong[] shape, ReadOnlyMemory<byte> data) {
            Dtype = dtype;
            Shape = shape;
            Data = data;
        }

        /// <summary>
        /// Converts the tensor data to an array of the specified type.
        /// </summary>
        /*public T[] ToArray<T>() where T : struct {
            int elementSize = sizeof(T);
            if (Data.Length % elementSize != 0)
                throw new InvalidOperationException("Data length is not a multiple of the element size.");

            int count = Data.Length / elementSize;
            T[] array = new T[count];
            Buffer.BlockCopy(Data.ToArray(), 0, array, 0, Data.Length);
            return array;
        }*/
    }

    /// <summary>
    /// Represents the entire safetensors file.
    /// </summary>
    public class SafeTensors : IDisposable {

        private const uint MAX_HEADER_SIZE = 100_000_000;

        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _headerAccessor;
        private readonly MemoryMappedViewAccessor _dataAccessor;

        public Dictionary<string, TensorInfo> TensorInfos { get; }
        public Dictionary<string, TensorView> Tensors { get; }

        public int Length { get { return Tensors.Count; } }

        private SafeTensors(string filePath, Dictionary<string, TensorInfo> tensorInfos, Dictionary<string, TensorView> tensors, MemoryMappedFile mmf, MemoryMappedViewAccessor headerAccessor, MemoryMappedViewAccessor dataAccessor) {
            _mmf = mmf;
            _headerAccessor = headerAccessor;
            _dataAccessor = dataAccessor;
            TensorInfos = tensorInfos;
            Tensors = tensors;
        }

        /// <summary>
        /// Deserializes a safetensors file.
        /// </summary>
        public static SafeTensors Deserialize(string filePath) {
            using FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (fs.Length < 8) {
                throw new HeaderTooSmallException();
            }

            using MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, false);

            // Read the first 8 bytes for header length
            using MemoryMappedViewAccessor headerAccessor = mmf.CreateViewAccessor(0, 8, MemoryMappedFileAccess.Read);
            byte[] array = new byte[8];
            headerAccessor.ReadArray(0, array, 0, 8);
            ulong headerLength = BitConverter.ToUInt64(array, 0);

            if (headerLength > MAX_HEADER_SIZE) {
                throw new HeaderTooLargeException();
            }

            ulong stop = CheckedAdd(headerLength, 8);
            if (stop > (ulong)fs.Length) {
                throw new InvalidHeaderLengthException();
            }

            // Read the header JSON string
            byte[] headerBytes = new byte[headerLength];
            using (MemoryMappedViewStream headerStream = mmf.CreateViewStream(8, (long)headerLength, MemoryMappedFileAccess.Read)) {
                headerStream.Read(headerBytes, 0, (int)headerLength);
            }

            string headerJson;
            try {
                var encoding = new UTF8Encoding(
                    encoderShouldEmitUTF8Identifier: false,
                    throwOnInvalidBytes: true
                );
                headerJson = encoding.GetString(headerBytes);
            } catch (Exception e) {
                throw new InvalidHeaderException(e);
            }

            // Deserialize metadata
            Dictionary<string, TensorInfo> tensorInfos;
            try {
                var metadata = JsonConvert.DeserializeObject<Dictionary<string, JToken>>(headerJson);

                // Parse tensor infos
                tensorInfos = new Dictionary<string, TensorInfo>();
                foreach (var kvp in metadata) {
                    tensorInfos[kvp.Key] = kvp.Value.ToObject<TensorInfo>();
                }
            } catch (Exception e) {
                throw new InvalidHeaderDeserializationException();
            }

            Metadata metadataInfo = new Metadata(tensorInfos);
            ulong bufferEnd = Validate(metadataInfo);
            if ((bufferEnd + 8 + headerLength) != (ulong)fs.Length) {
                throw new MetadataIncompleteBufferException();
            }

            // Initialize tensor views
            var tensors = new Dictionary<string, TensorView>();
            foreach (var kvp in tensorInfos) {
                string tensorName = kvp.Key;
                TensorInfo info = kvp.Value;

                ulong start = info.DataOffsets[0];
                ulong end = info.DataOffsets[1];
                ulong dataLength = end - start;

                // Create a view for the tensor data
                ReadOnlyMemory<byte> data = new ReadOnlyMemory<byte>(ReadData(mmf, (long)start + 8 + (long)headerLength, dataLength));

                tensors[tensorName] = new TensorView(info.Dtype, info.Shape, data);
            }

            return new SafeTensors(filePath, tensorInfos, tensors, mmf, headerAccessor, null);
        }

        private static ulong CheckedAdd(ulong a, ulong b) {
            try {
                return checked(a + b);
            } catch (OverflowException) {
                throw new InvalidHeaderLengthException();
            }
        }

        private static ulong Validate(Metadata metadata) {
            ulong start = 0;

            int i = 0;
            foreach (var kvp in metadata.Tensors) {
                TensorInfo info = kvp.Value;
                ulong s = info.DataOffsets[0];
                ulong e = info.DataOffsets[1];

                if (s != start || e < s) {
                    var tensorName = metadata.IndexMap.FirstOrDefault(kvp => kvp.Value == i).Key ?? "no_tensor";
                    throw new InvalidOffsetException(tensorName);
                }

                start = e;

                ulong nelements = 1;
                foreach (var dim in info.Shape) {
                    try {
                        nelements = checked(nelements * dim);
                    } catch (OverflowException) {
                        throw new ValidationOverflowException();
                    }
                }

                // Calculate the number of bytes
                ulong nbytes;
                try {
                    nbytes = checked(nelements * (ulong)info.Dtype.Size());
                } catch (OverflowException) {
                    throw new ValidationOverflowException();
                }

                if ((e - s) != nbytes) {
                    throw new TensorInvalidInfoException();
                }

                i++;
            }

            return start;
        }

        private static byte[] ReadData(MemoryMappedFile mmf, long position, ulong length) {
            byte[] data = new byte[length];
            using MemoryMappedViewStream dataStream = mmf.CreateViewStream(position, (long)length, MemoryMappedFileAccess.Read);
            dataStream.Read(data, 0, (int)length);
            return data;
        }

        /// <summary>
        /// Retrieves a tensor by name.
        /// </summary>
        public TensorView GetTensor(string name) {
            if (Tensors.TryGetValue(name, out TensorView tensor)) {
                return tensor;
            }
            throw new KeyNotFoundException($"Tensor '{name}' not found.");
        }

        public void Dispose() {
            _headerAccessor?.Dispose();
            _dataAccessor?.Dispose();
            _mmf?.Dispose();
        }
    }
}
