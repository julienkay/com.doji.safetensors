using NUnit.Framework;
using System.IO;
using System.Text;

namespace Doji.AI.Safetensors.Editor.Tests {

    public class SafetensorsTests {

        private static readonly string TmpPath = Path.Combine("Temp", "tmp.safetensors");

        private static void WriteFile(string serializedData) {
            byte[] byteArray = Encoding.UTF8.GetBytes(serializedData);
            File.WriteAllBytes(TmpPath, byteArray);
        }

        private static void Deserialize() {
            SafeTensors.Deserialize(TmpPath);
        }

        [Test]
        public void TestEmptyShapesAllowed() {
            WriteFile("8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00");
            var loaded = SafeTensors.Deserialize(TmpPath);
            Assert.AreEqual(new string[] { "test" }, loaded.Tensors.Keys);
            var tensor = loaded.Tensors["test"];
            Assert.IsEmpty(tensor.Shape);
            Assert.AreEqual(Dtype.I32, tensor.Dtype);
            Assert.AreEqual(new byte[] { 0, 0, 0, 0 }, tensor.Data.ToArray());
        }

        [Test]
        public void TestDeserialization() {
            WriteFile("<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
            var loaded = SafeTensors.Deserialize(TmpPath);
            Assert.AreEqual(1, loaded.Length);
            Assert.AreEqual(new string[] { "test" }, loaded.Tensors.Keys);
            var tensor = loaded.Tensors["test"];
            Assert.AreEqual(new ulong[] { 2, 2 }, tensor.Shape);
            Assert.AreEqual(Dtype.I32, tensor.Dtype);
            Assert.AreEqual(new byte[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, tensor.Data.ToArray());
        }

        [Test]
        public void TestMetadataIncomplete() {
            WriteFile("<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file");
            Assert.Throws<MetadataIncompleteBufferException>(Deserialize);
        }

        [Test]
        public void TestMetadataMissing() {
            WriteFile("<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"); // <--- missing 2 bytes);
            Assert.Throws<MetadataIncompleteBufferException>(Deserialize);
        }

        [Test]
        public void TestHeaderTooLarge() {
            WriteFile("<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
            Assert.Throws<HeaderTooLargeException>(Deserialize);
        }

        [Test]
        public void TestHeaderTooSmall() {
            WriteFile("");
            Assert.Throws<HeaderTooSmallException>(Deserialize);
        }

        [Test]
        public void TestInvalidHeaderLength() {
            WriteFile("<\x00\x00\x00\x00\x00\x00\x00");
            Assert.Throws<InvalidHeaderLengthException>(Deserialize);
        }

        [Test]
        public void TestInvalidHeaderNonUTF8() {
            WriteFile("\x01\x00\x00\x00\x00\x00\x00\x00\xff");
            Assert.Throws<InvalidHeaderException>(Deserialize);
        }

        [Test]
        public void TestInvalidHeaderNotJson() {
            WriteFile("\x01\x00\x00\x00\x00\x00\x00\x00{");
            Assert.Throws<InvalidHeaderDeserializationException>(Deserialize);
        }

        [Test]
        /// Test that the JSON header may be trailing-padded with JSON whitespace characters.
        public void TestWhitespacePaddedHeader() {
            WriteFile("\x06\x00\x00\x00\x00\x00\x00\x00{}\x0D\x20\x09\x0A");
            var loaded = SafeTensors.Deserialize(TmpPath);
            //assert_eq!(loaded.len(), 0);
        }

        [Test]
        public void TestZeroSizedTensor() {
            WriteFile("<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}");
            var loaded = SafeTensors.Deserialize(TmpPath);
            Assert.AreEqual(new string[] { "test" }, loaded.Tensors.Keys);
            var tensor = loaded.Tensors["test"];
            Assert.AreEqual(new ulong[] { 2, 0 }, tensor.Shape);
            Assert.AreEqual(Dtype.I32, tensor.Dtype);
            Assert.AreEqual(new byte[] { }, tensor.Data.ToArray());
        }

        [Test]
        public void TestInvalidInfo() {
            WriteFile("<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0, 4]}}");
            Assert.Throws<TensorInvalidInfoException>(Deserialize);
        }

        [Test]
        public void TestValidationOverflow1() {
            // u64::MAX =  18_446_744_073_709_551_615u64
            // Overflow the shape calculation.
            WriteFile("O\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
            Assert.Throws<ValidationOverflowException>(Deserialize);
        }

        [Test]
        public void TestValidationOverflow2() {
            // u64::MAX =  18_446_744_073_709_551_615u64
            // Overflow the num_elements * total shape.
            WriteFile("N\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
            Assert.Throws<ValidationOverflowException>(Deserialize);
        }
    }
}