// Run with:
// swiftc gary4local/gary4local/CareyModelInventory.swift scripts/tests/CareyModelInventoryTests.swift -o /tmp/carey-inventory-tests
// /tmp/carey-inventory-tests
import Foundation

// Foundation-only fixture for the target identity contract. The real enum
// lives in the SwiftUI view model; the app build verifies that integration.
enum CareyDownloadTarget: String {
    case base = "acestep-v15-base", sft = "acestep-v15-sft", turbo = "acestep-v15-turbo"
    case xlBase = "acestep-v15-xl-base", xlSft = "acestep-v15-xl-sft", xlTurbo = "acestep-v15-xl-turbo"
    case mlxBase = "mlx-acestep-v15-base", mlxSft = "mlx-acestep-v15-sft", mlxTurbo = "mlx-acestep-v15-turbo"
    case mlxXlBase = "mlx-acestep-v15-xl-base", mlxXlSft = "mlx-acestep-v15-xl-sft", mlxXlTurbo = "mlx-acestep-v15-xl-turbo"
    case captionerSmall, captionerMedium, captionerLarge
    var careyConfigName: String? { rawValue.replacingOccurrences(of: "mlx-", with: "") }
    static let ditTargets: [Self] = [.base, .sft, .turbo, .xlBase, .xlSft, .xlTurbo]
    static let optimizedTargets: [Self] = [.mlxBase, .mlxSft, .mlxTurbo, .mlxXlBase, .mlxXlSft, .mlxXlTurbo]
}

@main
struct CareyModelInventoryTests {
    static func main() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("carey-inventory-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let checkpoints = root.appendingPathComponent("checkpoints")
        try fm.createDirectory(at: checkpoints, withIntermediateDirectories: true)
        func put(_ path: String, _ contents: String = "fixture") throws {
            let url = checkpoints.appendingPathComponent(path)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(contents.utf8).write(to: url)
        }

        precondition(!CareyModelInventory.hasLegacyWeights(in: checkpoints), "fresh install")
        try put("Qwen3-Embedding-0.6B/model.safetensors")
        try put("vae/model.safetensors")
        try put("acestep-v15-base/config.json")
        try put("acestep-v15-base/silence_latent.pt")
        try put("acestep-v15-base/model.safetensors.index.json")
        try put("acestep-v15-base/custom-adapter.safetensors")
        precondition(!CareyModelInventory.hasLegacyWeights(in: checkpoints), "shared assets and retained metadata aren't legacy DiTs")
        try put("acestep-v15-base/model.safetensors", "")
        precondition(!CareyModelInventory.hasLegacyWeights(in: checkpoints), "empty downloads don't count")
        try put("acestep-v15-base/model.safetensors")
        precondition(CareyModelInventory.hasLegacyWeights(in: checkpoints), "single-file legacy weights")
        try put("acestep-v15-xl-base/model-00001-of-00004.safetensors")
        try fm.removeItem(at: checkpoints.appendingPathComponent("acestep-v15-base/model.safetensors"))
        precondition(CareyModelInventory.hasLegacyWeights(in: checkpoints), "partial XL shard set stays visible")
        try fm.removeItem(at: checkpoints.appendingPathComponent("acestep-v15-xl-base/model-00001-of-00004.safetensors"))
        precondition(!CareyModelInventory.hasLegacyWeights(in: checkpoints), "last legacy model removal")

        let blob = root.appendingPathComponent("blob")
        try Data("weights".utf8).write(to: blob)
        let legacyLink = checkpoints.appendingPathComponent("acestep-v15-base/model.safetensors")
        try fm.createSymbolicLink(at: legacyLink, withDestinationURL: blob)
        precondition(CareyModelInventory.hasLegacyWeights(in: checkpoints), "HF symlinks count")
        try fm.removeItem(at: legacyLink)

        let hub = root.appendingPathComponent("hub")
        let artifact = hub.appendingPathComponent("models--thepatch--acestep-1.5-mlx/snapshots/\(CareyOptimizedModelInventory.revision)/MLX/acestep-v15-base-f16")
        try fm.createDirectory(at: artifact.appendingPathComponent("conditioning"), withIntermediateDirectories: true)
        let provenance = #"{"variant":"acestep-v15-base","source":{"revision":"e432212fec32b8965a14ffa57ae653438d6abd14"},"artifact":{"files":[{"filename":"model-00001.safetensors"}]}}"#
        for file in ["provenance.json", "conditioning/provenance.json"] {
            try Data(provenance.utf8).write(to: artifact.appendingPathComponent(file))
        }
        for file in ["config.json", "model.safetensors.index.json", "conditioning/model.safetensors.index.json", "conditioning/silence_latent.pt"] {
            try Data("fixture".utf8).write(to: artifact.appendingPathComponent(file))
        }
        for file in ["model-00001.safetensors", "conditioning/model-00001.safetensors"] {
            try fm.createSymbolicLink(at: artifact.appendingPathComponent(file), withDestinationURL: blob)
        }
        let optimized = CareyOptimizedModelInventory.statuses(hubCacheDirectory: hub)
        precondition(optimized.filter(\.downloaded).map(\.target) == [.mlxBase], "complete artifact with HF links")
        precondition(!CareyModelInventory.hasLegacyWeights(in: checkpoints), "optimized-only install hides legacy catalog")
        precondition(CareyModelInventory.downloadedTrainingModels(in: checkpoints, optimizedModels: optimized) == [.base], "optimized-only training availability")
        try fm.removeItem(at: artifact.appendingPathComponent("conditioning/model-00001.safetensors"))
        let incomplete = CareyOptimizedModelInventory.statuses(hubCacheDirectory: hub)
        precondition(!incomplete.contains(where: \.downloaded), "partial optimized artifact isn't ready")
        precondition(CareyModelInventory.downloadedTrainingModels(in: checkpoints, optimizedModels: incomplete).isEmpty, "partial artifact can't train")
        print("Carey inventory fixture assertions passed (fresh, legacy, partial, symlink, optimized-only, last removal).")
    }
}
