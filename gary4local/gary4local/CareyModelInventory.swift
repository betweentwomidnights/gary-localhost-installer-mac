import Foundation

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
