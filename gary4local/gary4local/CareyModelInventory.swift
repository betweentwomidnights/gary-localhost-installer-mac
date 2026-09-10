import Foundation

struct CareyOptimizedModelStatus: Identifiable {
    let target: CareyDownloadTarget
    let downloaded: Bool
    let sizeBytes: Int64

    var id: String { target.rawValue }
}

enum CareyOptimizedModelInventory {
    static let revision = "218ca5e51809821177bdfed70bf500020df6e958"
    private static let repositoryCacheName = "models--thepatch--acestep-1.5-mlx"

    static func statuses(hubCacheDirectory: URL) -> [CareyOptimizedModelStatus] {
        CareyDownloadTarget.optimizedTargets.map { target in
            let root = artifactRoot(for: target, hubCacheDirectory: hubCacheDirectory)
            let size = validatedSize(of: root, variant: target.careyConfigName ?? "")
            return CareyOptimizedModelStatus(
                target: target,
                downloaded: size != nil,
                sizeBytes: size ?? 0
            )
        }
    }

    private static func artifactRoot(
        for target: CareyDownloadTarget,
        hubCacheDirectory: URL
    ) -> URL {
        let variant = target.careyConfigName ?? ""
        return hubCacheDirectory
            .appendingPathComponent(repositoryCacheName, isDirectory: true)
            .appendingPathComponent("snapshots/\(revision)/MLX", isDirectory: true)
            .appendingPathComponent("\(variant)-f16", isDirectory: true)
    }

    private static func validatedSize(of root: URL, variant: String) -> Int64? {
        let expectedSourceRevisions = [
            "acestep-v15-base": "e432212fec32b8965a14ffa57ae653438d6abd14",
            "acestep-v15-sft": "c410d249e71ea9385a7b586865e65b1473e1098d",
            "acestep-v15-turbo": "19671f406d603126926c1b7e2adc169acbcade22",
            "acestep-v15-xl-base": "220c1166efbdd9583eafcb12eb160594bbfcb241",
            "acestep-v15-xl-sft": "d06de46b4622f781cf07f4a013a67d591ca52819",
            "acestep-v15-xl-turbo": "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
        ]
        guard let expectedSourceRevision = expectedSourceRevisions[variant] else { return nil }
        guard let decoder = provenance(at: root.appendingPathComponent("provenance.json")),
              decoder.variant == variant,
              decoder.source.revision == expectedSourceRevision,
              let conditioning = provenance(
                at: root.appendingPathComponent("conditioning/provenance.json")
              ),
              conditioning.variant == variant,
              conditioning.source.revision == expectedSourceRevision else {
            return nil
        }

        let fixed = [
            root.appendingPathComponent("config.json"),
            root.appendingPathComponent("model.safetensors.index.json"),
            root.appendingPathComponent("conditioning/model.safetensors.index.json"),
            root.appendingPathComponent("conditioning/silence_latent.pt"),
        ]
        let recorded = decoder.artifact.files.map { root.appendingPathComponent($0.filename) }
            + conditioning.artifact.files.map {
                root.appendingPathComponent("conditioning/\($0.filename)")
            }
        let paths = fixed + recorded
        guard paths.allSatisfy({ nonEmptyFileSize($0) != nil }) else { return nil }
        return paths.compactMap(nonEmptyFileSize).reduce(0, +)
    }

    private static func provenance(at url: URL) -> ArtifactProvenance? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ArtifactProvenance.self, from: data)
    }

    private static func nonEmptyFileSize(_ url: URL) -> Int64? {
        let resolvedURL = url.resolvingSymlinksInPath()
        guard let values = try? resolvedURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              let size = values.fileSize,
              size > 0 else {
            return nil
        }
        return Int64(size)
    }

    private struct ArtifactProvenance: Decodable {
        let variant: String
        let source: Source
        let artifact: Artifact
    }

    private struct Source: Decodable {
        let revision: String?
    }

    private struct Artifact: Decodable {
        let files: [ArtifactFile]
    }

    private struct ArtifactFile: Decodable {
        let filename: String
    }
}

/// What "downloaded" means for the Carey checkpoints, in one place.
///
/// The file lists here mirror `scripts/download_carey_models.sh` -- that script
/// writes them, this reads them back. Keep the two in step: a model whose files
/// are listed differently in the two places either looks missing after a
/// successful download, or looks present when it is half there.
///
/// Everything is answered against a checkpoints directory (the one
/// `ACESTEP_CHECKPOINT_DIR` points at), so the trainers can ask without holding
/// a reference to the view model.

// MARK: - Captioner LMs

enum CareyCaptionerModel: String, CaseIterable, Identifiable, Codable {
    case small = "acestep-5Hz-lm-0.6B"
    case medium = "acestep-5Hz-lm-1.7B"
    case large = "acestep-5Hz-lm-4B"

    var id: String { rawValue }

    /// What the trainers and sa3_autolabel.py take for --caption-lm-model.
    var argument: String { rawValue }

    static let fallback: CareyCaptionerModel = .medium

    var shortName: String {
        switch self {
        case .small: return "0.6B"
        case .medium: return "1.7B"
        case .large: return "4B"
        }
    }

    /// Matches the wording the Carey trainer already uses.
    var menuLabel: String {
        switch self {
        case .small: return "0.6B · not recommended"
        case .medium: return "1.7B · good quality"
        case .large: return "4B · best quality, large memory"
        }
    }

    var downloadTarget: CareyDownloadTarget {
        switch self {
        case .small: return .captionerSmall
        case .medium: return .captionerMedium
        case .large: return .captionerLarge
        }
    }

    /// Paths relative to the checkpoints directory.
    var requiredFiles: [String] {
        // 4B ships as two shards plus an index; the others are a single file.
        let weights: [String] = self == .large
            ? [
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
            : ["model.safetensors"]
        let common = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ]
        return (common + weights).map { "\(rawValue)/\($0)" }
    }
}

// MARK: - Training / preparation models

enum CareyTrainingModel: String, CaseIterable, Identifiable, Codable {
    case base = "acestep-v15-base"
    case xlBase = "acestep-v15-xl-base"

    var id: String { rawValue }

    /// What train_mlx_lora_job.py takes for --model. MODEL_MAP is keyed by
    /// these aliases, not by the folder names.
    var argument: String {
        switch self {
        case .base: return "base"
        case .xlBase: return "xl-base"
        }
    }

    var menuLabel: String {
        switch self {
        case .base: return "ace-step-v15-base"
        case .xlBase: return "ace-step-v15-xl-base"
        }
    }

    /// Paths relative to the checkpoints directory. `silence_latent.pt` is in
    /// here deliberately: captioning and training both need it, and a model
    /// missing only that file used to train against another model's latent.
    var requiredFiles: [String] {
        let weights: [String] = self == .xlBase
            ? [
                "model.safetensors.index.json",
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ]
            : ["model.safetensors"]
        return (["config.json"] + weights + ["silence_latent.pt"]).map { "\(rawValue)/\($0)" }
    }
}

// MARK: - Inventory

enum CareyModelInventory {
    /// A file counts as present only when it exists and is non-empty. An
    /// interrupted download can leave a zero-byte file behind, and treating
    /// that as downloaded is how you get a failure several minutes into a run.
    static func isFilePresent(_ relativePath: String, in checkpointDirectory: URL) -> Bool {
        let url = checkpointDirectory
            .appendingPathComponent(relativePath)
            .standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else {
            return false
        }
        return size.int64Value > 0
    }

    static func missingFiles(
        _ relativePaths: [String],
        in checkpointDirectory: URL
    ) -> [String] {
        relativePaths.filter { !isFilePresent($0, in: checkpointDirectory) }
    }

    static func isDownloaded(_ model: CareyCaptionerModel, in checkpointDirectory: URL) -> Bool {
        missingFiles(model.requiredFiles, in: checkpointDirectory).isEmpty
    }

    static func isDownloaded(_ model: CareyTrainingModel, in checkpointDirectory: URL) -> Bool {
        missingFiles(model.requiredFiles, in: checkpointDirectory).isEmpty
    }

    static func downloadedCaptioners(in checkpointDirectory: URL) -> [CareyCaptionerModel] {
        CareyCaptionerModel.allCases.filter { isDownloaded($0, in: checkpointDirectory) }
    }

    static func downloadedTrainingModels(in checkpointDirectory: URL) -> [CareyTrainingModel] {
        CareyTrainingModel.allCases.filter { isDownloaded($0, in: checkpointDirectory) }
    }

    /// Pick a model that is actually present, preferring the caller's choice.
    /// A saved selection can point at something removed since it was made.
    static func resolvedCaptioner(
        preferring requested: CareyCaptionerModel,
        in checkpointDirectory: URL
    ) -> CareyCaptionerModel? {
        if isDownloaded(requested, in: checkpointDirectory) { return requested }
        return downloadedCaptioners(in: checkpointDirectory).first
    }

    static func resolvedTrainingModel(
        preferring requested: CareyTrainingModel,
        in checkpointDirectory: URL
    ) -> CareyTrainingModel? {
        if isDownloaded(requested, in: checkpointDirectory) { return requested }
        return downloadedTrainingModels(in: checkpointDirectory).first
    }
}
