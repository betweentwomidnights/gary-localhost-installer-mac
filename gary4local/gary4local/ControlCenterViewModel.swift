import Foundation
import SwiftUI
import Combine
import AppKit
import Darwin

enum StableAudioBackendEngine: String, CaseIterable, Identifiable {
    case mps
    case mlx

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .mps:
            return "mps"
        case .mlx:
            return "mlx"
        }
    }

    static func from(rawValue: String) -> StableAudioBackendEngine {
        StableAudioBackendEngine(rawValue: rawValue.lowercased()) ?? .mps
    }
}

enum MelodyFlowBackendEngine: String, CaseIterable, Identifiable {
    case mps
    case mlxNativeTorchCodec = "mlx_native_torch_codec"
    case mlxNativeMlxCodec = "mlx_native_mlx_codec"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .mps:
            return "mps"
        case .mlxNativeTorchCodec:
            return "mlx + torch codec"
        case .mlxNativeMlxCodec:
            return "mlx end-to-end"
        }
    }

    var shortDisplayName: String {
        switch self {
        case .mps:
            return "mps"
        case .mlxNativeTorchCodec:
            return "mlx+torch"
        case .mlxNativeMlxCodec:
            return "mlx e2e"
        }
    }

    static func from(rawValue: String) -> MelodyFlowBackendEngine {
        MelodyFlowBackendEngine(rawValue: rawValue.lowercased()) ?? .mps
    }
}

enum CareyBackendEngine: String, CaseIterable, Identifiable {
    case mps
    case mlx

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .mps:
            return "mps"
        case .mlx:
            return "mlx"
        }
    }

    static func from(rawValue: String) -> CareyBackendEngine {
        CareyBackendEngine(rawValue: rawValue.lowercased()) ?? .mlx
    }
}

struct SA3RuntimeSettings: Codable, Equatable {
    var peakNormalizeDb: String
    var limiterCeilingDb: String
    var latentRescale: String
    var latentShift: String
    var latentTargetStd: String
    var continuationTailPad: String
    var useFP32DiT: Bool

    static let fallbackDefaults = SA3RuntimeSettings(
        peakNormalizeDb: "2.0",
        limiterCeilingDb: "-0.3",
        latentRescale: "1.0",
        latentShift: "0.0",
        latentTargetStd: "",
        continuationTailPad: "6",
        useFP32DiT: true
    )

    init(
        peakNormalizeDb: String,
        limiterCeilingDb: String,
        latentRescale: String,
        latentShift: String,
        latentTargetStd: String,
        continuationTailPad: String,
        useFP32DiT: Bool
    ) {
        self.peakNormalizeDb = peakNormalizeDb
        self.limiterCeilingDb = limiterCeilingDb
        self.latentRescale = latentRescale
        self.latentShift = latentShift
        self.latentTargetStd = latentTargetStd
        self.continuationTailPad = continuationTailPad
        self.useFP32DiT = useFP32DiT
    }

    init(environment: [String: String]) {
        let defaults = Self.fallbackDefaults
        peakNormalizeDb = environment["SA3_PEAK_NORMALIZE_DB"] ?? defaults.peakNormalizeDb
        limiterCeilingDb = environment["SA3_LIMITER_CEILING_DB"] ?? defaults.limiterCeilingDb
        latentRescale = environment["SA3_LATENT_RESCALE"] ?? defaults.latentRescale
        latentShift = environment["SA3_LATENT_SHIFT"] ?? defaults.latentShift
        latentTargetStd = environment["SA3_LATENT_TARGET_STD"] ?? defaults.latentTargetStd
        continuationTailPad = environment["SA3_CONTINUE_TAIL_PAD"] ?? defaults.continuationTailPad
        useFP32DiT = (environment["SA3_MLX_DIT_DTYPE"] ?? "float32").lowercased() == "float32"
    }

    var normalized: SA3RuntimeSettings {
        SA3RuntimeSettings(
            peakNormalizeDb: peakNormalizeDb.trimmingCharacters(in: .whitespacesAndNewlines),
            limiterCeilingDb: limiterCeilingDb.trimmingCharacters(in: .whitespacesAndNewlines),
            latentRescale: latentRescale.trimmingCharacters(in: .whitespacesAndNewlines),
            latentShift: latentShift.trimmingCharacters(in: .whitespacesAndNewlines),
            latentTargetStd: latentTargetStd.trimmingCharacters(in: .whitespacesAndNewlines),
            continuationTailPad: continuationTailPad.trimmingCharacters(in: .whitespacesAndNewlines),
            useFP32DiT: useFP32DiT
        )
    }
}

struct DownloadableModel: Identifiable {
    let id: String
    let size: String
    let displayName: String
    let path: String
    var downloaded: Bool
    var isDownloading: Bool
    var progress: Double
    var statusMessage: String
}

struct DownloadModelSection: Identifiable {
    let id: String
    let title: String
    let models: [DownloadableModel]
}

struct StableAudioInventoryModelStatus: Identifiable {
    let id: String
    let label: String
    let downloaded: Bool
    let missing: [String]
}

struct CareyRequiredModelStatus: Identifiable {
    let id: String
    let label: String
    let relativePath: String
    let downloaded: Bool
    let sizeBytes: Int64
}

enum CareyDownloadTarget: String, CaseIterable, Identifiable {
    case base
    case sft
    case turbo
    case shared
    case scragVae = "scrag-vae"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .base:
            return "base"
        case .sft:
            return "sft"
        case .turbo:
            return "turbo"
        case .shared:
            return "shared"
        case .scragVae:
            return "scrag vae"
        }
    }
}

enum CareyLoraModelFamily: String, CaseIterable, Identifiable, Codable {
    case standard
    case xl

    var id: String { rawValue }
}

struct CareyLoraCatalogEntry: Codable {
    var path: String
    var captionsPath: String?
    var scale: Double
    var backends: [String]
    var modelFamily: String

    init(
        path: String,
        captionsPath: String?,
        scale: Double = 1.0,
        backends: [String] = ["base", "turbo"],
        modelFamily: String = "standard"
    ) {
        self.path = path
        self.captionsPath = captionsPath
        self.scale = scale
        self.backends = backends
        self.modelFamily = modelFamily
    }

    enum CodingKeys: String, CodingKey {
        case path
        case captionsPath
        case scale
        case backends
        case modelFamily
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        path = try container.decode(String.self, forKey: .path)
        captionsPath = try container.decodeIfPresent(String.self, forKey: .captionsPath)
        scale = try container.decodeIfPresent(Double.self, forKey: .scale) ?? 1.0
        backends = try container.decodeIfPresent([String].self, forKey: .backends) ?? ["base", "turbo"]
        modelFamily = try container.decodeIfPresent(String.self, forKey: .modelFamily) ?? "standard"
    }
}

struct CareyLoraEntry: Identifiable {
    let name: String
    let path: String
    let captionsPath: String?
    let resolvedCaptionsPath: String?
    let captionCount: Int
    let scale: Double
    let backends: [String]
    let modelFamily: String
    let checkpointExists: Bool
    let registered: Bool

    var id: String { name }
}

struct CareyLoraState {
    let entries: [CareyLoraEntry]
    let pools: [String: Int]
    let catalogPath: String
    let registryPath: String
    let captionsPath: String
}

struct SA3InventoryModelStatus: Identifiable {
    let repoID: String
    let label: String
    let downloaded: Bool
    let missing: [String]

    var id: String { "\(repoID)::\(label)" }
}

struct SA3TrainingCheckpoint: Codable, Identifiable, Equatable {
    var step: Int
    var epoch: Int
    var path: String

    var id: String { "\(step)::\(epoch)::\(path)" }

    enum CodingKeys: String, CodingKey {
        case step
        case epoch
        case path
    }

    init(step: Int, epoch: Int = 0, path: String) {
        self.step = step
        self.epoch = epoch
        self.path = path
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        step = try container.decode(Int.self, forKey: .step)
        epoch = try container.decodeIfPresent(Int.self, forKey: .epoch) ?? 0
        path = try container.decode(String.self, forKey: .path)
    }
}

struct SA3LoraCatalogEntry: Codable {
    var path: String
    var promptsPath: String?
    var strength: Double
    var trainingBaseModel: String?
    var inferenceModel: String?
    var trainingJobId: String?
    var trainingCheckpoints: [SA3TrainingCheckpoint]
    var selectedTrainingStep: Int?

    init(
        path: String,
        promptsPath: String?,
        strength: Double = 1.0,
        trainingBaseModel: String? = nil,
        inferenceModel: String? = nil,
        trainingJobId: String? = nil,
        trainingCheckpoints: [SA3TrainingCheckpoint] = [],
        selectedTrainingStep: Int? = nil
    ) {
        self.path = path
        self.promptsPath = promptsPath
        self.strength = strength
        self.trainingBaseModel = trainingBaseModel
        self.inferenceModel = inferenceModel
        self.trainingJobId = trainingJobId
        self.trainingCheckpoints = trainingCheckpoints
        self.selectedTrainingStep = selectedTrainingStep
    }

    enum CodingKeys: String, CodingKey {
        case path
        case promptsPath
        case strength
        case trainingBaseModel
        case inferenceModel
        case trainingJobId
        case trainingCheckpoints
        case selectedTrainingStep
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        path = try container.decode(String.self, forKey: .path)
        promptsPath = try container.decodeIfPresent(String.self, forKey: .promptsPath)
        strength = try container.decodeIfPresent(Double.self, forKey: .strength) ?? 1.0
        trainingBaseModel = try container.decodeIfPresent(
            String.self,
            forKey: .trainingBaseModel
        )
        inferenceModel = try container.decodeIfPresent(
            String.self,
            forKey: .inferenceModel
        )
        trainingJobId = try container.decodeIfPresent(
            String.self,
            forKey: .trainingJobId
        )
        trainingCheckpoints = try container.decodeIfPresent(
            [SA3TrainingCheckpoint].self,
            forKey: .trainingCheckpoints
        ) ?? []
        selectedTrainingStep = try container.decodeIfPresent(
            Int.self,
            forKey: .selectedTrainingStep
        )
    }
}

struct SA3LoraEntry: Identifiable {
    let name: String
    let path: String
    let promptsPath: String?
    let resolvedPromptsPath: String?
    let promptFilePath: String
    let promptFileExists: Bool
    let promptCount: Int
    let captionCount: Int
    let strength: Double
    let checkpointExists: Bool
    let registered: Bool
    let trainingJobId: String?
    let trainingCheckpoints: [SA3TrainingCheckpoint]
    let selectedTrainingStep: Int?

    var id: String { name }
}

struct SA3LoraState {
    let entries: [SA3LoraEntry]
    let pools: [String: Int]
    let catalogPath: String
    let registryPath: String
    let promptsDir: String
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

private extension Optional where Wrapped == String {
    var nilIfEmpty: String? {
        switch self {
        case .some(let value):
            return value.isEmpty ? nil : value
        case .none:
            return nil
        }
    }
}

@MainActor
final class ControlCenterViewModel: ObservableObject {
    private static let stableAudioBackendDefaultsKey = "stableAudioBackendEngine"
    private static let melodyFlowBackendDefaultsKey = "melodyFlowBackendEngine"
    private static let careyBackendDefaultsKey = "careyBackendEngine"
    private static let careyUseXlModelsDefaultsKey = "careyUseXlModels"
    private static let careyUseScragVaeDefaultsKey = "careyUseScragVae"
    private static let careyUseSampledMlxVaeEncodeDefaultsKey = "careyUseSampledMlxVaeEncode"
    private static let experimentalCareyMlxVaeEncodeToggleDefaultsKey = "experimentalCareyMlxVaeEncodeToggleEnabled"
    private static let sa3RuntimeSettingsDefaultsKey = "sa3RuntimeSettings"

    @Published var manager: ServiceManager?
    @Published var startupError: String?
    @Published var manifestPath: String = ""
    @Published var selectedServiceID: String?
    @Published var selectedLogText: String = ""
    @Published var isLogViewerPinnedToBottom: Bool = true
    @Published var stableAudioTokenInput: String = ""
    @Published var stableAudioTokenConfigured: Bool = false
    @Published var stableAudioTokenStatus: String = ""
    @Published var stableAudioStep2ScreenshotPath: String?
    @Published var isModelDownloadSheetPresented: Bool = false
    @Published var downloadableModels: [DownloadableModel] = []
    @Published var modelDownloadStatusMessage: String = ""
    @Published var isModelCatalogLoading: Bool = false
    @Published var isModelDownloadInProgress: Bool = false
    @Published var stableAudioPredownloadRepoInput: String = "thepatch/jerry_grunge"
    @Published var stableAudioPredownloadCheckpoints: [String] = []
    @Published var stableAudioPredownloadCheckpointDownloaded: [String: Bool] = [:]
    @Published var stableAudioInventoryModels: [StableAudioInventoryModelStatus] = []
    @Published var stableAudioCachedFinetunes: [String] = []
    @Published var stableAudioPredownloadSelectedCheckpoint: String = ""
    @Published var isStableAudioCheckpointFetchInProgress: Bool = false
    @Published var isStableAudioModelSwitchInProgress: Bool = false
    @Published var stableAudioPredownloadProgress: Double = 0
    @Published var stableAudioPredownloadTargetLabel: String = ""
    @Published var sa3InventoryModels: [SA3InventoryModelStatus] = []
    @Published var sa3PredownloadProgress: Double = 0
    @Published var sa3PredownloadTargetLabel: String = ""
    @Published var careyRequiredModels: [CareyRequiredModelStatus] = []
    @Published var careyOptionalModels: [CareyRequiredModelStatus] = []
    @Published var isCareyDownloadInProgress: Bool = false
    @Published var isCareyLifecycleActionInProgress: Bool = false
    @Published var careyPredownloadProgress: Double = 0
    @Published var careyPredownloadActiveLabel: String = ""
    @Published var stableAudioBackendEngine: StableAudioBackendEngine = .mps
    @Published var melodyFlowBackendEngine: MelodyFlowBackendEngine = .mps
    @Published var careyBackendEngine: CareyBackendEngine = .mlx
    @Published var careyUseXlModels: Bool = false
    @Published var careyUseScragVae: Bool = false
    @Published var careyUseSampledMlxVaeEncode: Bool = true
    @Published var sa3PeakNormalizeDb: String = SA3RuntimeSettings.fallbackDefaults.peakNormalizeDb
    @Published var sa3LimiterCeilingDb: String = SA3RuntimeSettings.fallbackDefaults.limiterCeilingDb
    @Published var sa3LatentRescale: String = SA3RuntimeSettings.fallbackDefaults.latentRescale
    @Published var sa3LatentShift: String = SA3RuntimeSettings.fallbackDefaults.latentShift
    @Published var sa3LatentTargetStd: String = SA3RuntimeSettings.fallbackDefaults.latentTargetStd
    @Published var sa3ContinuationTailPad: String = SA3RuntimeSettings.fallbackDefaults.continuationTailPad
    @Published var sa3UseFP32DiT: Bool = SA3RuntimeSettings.fallbackDefaults.useFP32DiT
    @Published var isCareyLoraSheetPresented: Bool = false
    @Published var careyLoraState: CareyLoraState?
    @Published var isCareyLoraLoading: Bool = false
    @Published var isCareyLoraSaving: Bool = false
    @Published var isCareyLoraBuilding: Bool = false
    @Published var careyLoraStatusMessage: String = ""
    @Published var careyLoraErrorMessage: String = ""
    @Published var careyLoraBuildOutput: String = ""
    @Published var isCareyAceTrainingSheetPresented: Bool = false
    @Published var isSA3LoraSheetPresented: Bool = false
    @Published var isSA3TrainingSheetPresented: Bool = false
    @Published var sa3LoraState: SA3LoraState?
    @Published var isSA3LoraLoading: Bool = false
    @Published var isSA3LoraSaving: Bool = false
    @Published var isSA3LoraBuilding: Bool = false
    @Published var sa3LoraSwitchingName: String?
    @Published var sa3LoraStatusMessage: String = ""
    @Published var sa3LoraErrorMessage: String = ""
    @Published var sa3LoraBuildOutput: String = ""
    @Published var melodyFlowBackendStatus: String = ""
    @Published var careyBackendStatus: String = ""
    @Published var sa3RuntimeSettingsStatus: String = ""
    @Published var rebuildFailureReport: RebuildFailureReport?
    @Published var rebuildDiagnosticsStatusMessage: String = ""
    @Published var isRequirementsEditorPresented: Bool = false
    @Published var requirementsEditorPath: String = ""
    @Published var requirementsEditorText: String = ""
    @Published var requirementsEditorStatusMessage: String = ""
    @Published var modelDownloadServiceID: String = "audiocraft_mlx"

    let sa3TrainingManager = SA3LoraTrainingManager()
    let careyAceTrainingManager = CareyAceTrainingManager()
    let sa3AutolabelManager = SA3AutolabelManager()

    private var logRefreshTask: Task<Void, Never>?
    private var modelDownloadPollTask: Task<Void, Never>?
    private var garyLocalDownloadTask: Task<Void, Never>?
    private var foundationLocalDownloadTask: Task<Void, Never>?
    private var melodyflowLocalDownloadTask: Task<Void, Never>?
    private var stableAudioLocalDownloadTask: Task<Void, Never>?
    private var sa3LocalDownloadTask: Task<Void, Never>?
    private var cancellables = Set<AnyCancellable>()
    private var managerCancellables = Set<AnyCancellable>()
    private var isLogRefreshInFlight = false
    private var pendingForcedLogRefresh = false
    private var lastLogMetadataByService: [String: LogMetadata] = [:]
    private let logRefreshIntervalNanoseconds: UInt64 = 300_000_000
    private let modelDownloadPollIntervalNanoseconds: UInt64 = 1_250_000_000
    private var activeModelDownloadPath: String?
    private var activeModelDownloadSessionID: String?
    private var careyDownloadTask: Task<Void, Never>?
    private var careyProgressByLabel: [String: Int] = [:]
    private var careyActiveDownloadTargets: [CareyDownloadTarget] = []
    private var sa3DefaultRuntimeSettings = SA3RuntimeSettings.fallbackDefaults
    private var sharedHuggingFaceToken: String?
    private var lastHandledSA3TrainingJobID: String?

    private static let careyProgressPercentRegex = try! NSRegularExpression(
        pattern: #"^[A-Za-z_]+:\s+([0-9]{1,3})%"#
    )

    static let experimentalCareyMlxVaeEncodeFeatureFlagKey = "experimentalCareyMlxVaeEncodeToggleEnabled"

    private static func isExperimentalCareyMlxVaeEncodeToggleEnabled() -> Bool {
        if UserDefaults.standard.object(forKey: experimentalCareyMlxVaeEncodeToggleDefaultsKey) != nil {
            return UserDefaults.standard.bool(forKey: experimentalCareyMlxVaeEncodeToggleDefaultsKey)
        }
        let environmentValue = ProcessInfo.processInfo.environment["GARY_EXPERIMENTAL_CAREY_MLX_VAE_ENCODE_TOGGLE"]
        guard let environmentValue else { return false }
        return ["1", "true", "yes", "on"].contains(environmentValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    private static let careySharedRequiredModelFiles: [(label: String, relativePath: String)] = [
        ("Qwen Weights", "checkpoints/Qwen3-Embedding-0.6B/model.safetensors"),
        ("Qwen Config", "checkpoints/Qwen3-Embedding-0.6B/config.json"),
        ("Qwen Tokenizer", "checkpoints/Qwen3-Embedding-0.6B/tokenizer.json"),
        ("Qwen Tokenizer Config", "checkpoints/Qwen3-Embedding-0.6B/tokenizer_config.json"),
        ("Qwen Merges", "checkpoints/Qwen3-Embedding-0.6B/merges.txt"),
        ("Qwen Vocab", "checkpoints/Qwen3-Embedding-0.6B/vocab.json"),
        ("Qwen Special Tokens", "checkpoints/Qwen3-Embedding-0.6B/special_tokens_map.json"),
        ("Qwen Added Tokens", "checkpoints/Qwen3-Embedding-0.6B/added_tokens.json"),
        ("Qwen Chat Template", "checkpoints/Qwen3-Embedding-0.6B/chat_template.jinja"),
        ("VAE Weights", "checkpoints/vae/diffusion_pytorch_model.safetensors"),
        ("VAE Config", "checkpoints/vae/config.json"),
    ]

    private static let careyOptionalModelFiles: [(label: String, relativePath: String)] = [
        ("ScragVAE Weights", "checkpoints/scrag-vae/diffusion_pytorch_model.safetensors"),
        ("ScragVAE Config", "checkpoints/scrag-vae/config.json"),
    ]

    private static func careyDiTRequiredFiles(
        labelPrefix: String,
        configName: String
    ) -> [(label: String, relativePath: String)] {
        let relativePrefix = "checkpoints/\(configName)"
        if configName.hasPrefix("acestep-v15-xl-") {
            return [
                ("\(labelPrefix) Config", "\(relativePrefix)/config.json"),
                ("\(labelPrefix) Weights Index", "\(relativePrefix)/model.safetensors.index.json"),
                ("\(labelPrefix) Weights Shard 1", "\(relativePrefix)/model-00001-of-00004.safetensors"),
                ("\(labelPrefix) Weights Shard 2", "\(relativePrefix)/model-00002-of-00004.safetensors"),
                ("\(labelPrefix) Weights Shard 3", "\(relativePrefix)/model-00003-of-00004.safetensors"),
                ("\(labelPrefix) Weights Shard 4", "\(relativePrefix)/model-00004-of-00004.safetensors"),
                ("\(labelPrefix) Silence Latent", "\(relativePrefix)/silence_latent.pt"),
            ]
        }

        return [
            ("\(labelPrefix) Config", "\(relativePrefix)/config.json"),
            ("\(labelPrefix) Weights", "\(relativePrefix)/model.safetensors"),
            ("\(labelPrefix) Silence Latent", "\(relativePrefix)/silence_latent.pt"),
        ]
    }

    private struct LogMetadata: Equatable {
        let fileSize: UInt64
        let modificationDate: Date?
    }

    private struct RemoteModelsResponse: Decodable {
        let success: Bool
        let models: [String: [RemoteModelEntry]]
    }

    private struct RemoteModelEntry: Decodable {
        let name: String
        let path: String?
        let type: String
        let checkpoints: [RemoteModelCheckpoint]?
    }

    private struct RemoteModelCheckpoint: Decodable {
        let name: String
        let path: String
        let epoch: Int?
    }

    private struct RemoteDownloadStatusResponse: Decodable {
        let success: Bool
        let models: [String: RemoteDownloadStatus]
    }

    private struct RemoteDownloadStatus: Decodable {
        let downloaded: Bool
        let missing: [String]?
    }

    private struct RemotePredownloadStartResponse: Decodable {
        let success: Bool
        let sessionID: String
        let modelName: String
        let message: String?

        enum CodingKeys: String, CodingKey {
            case success
            case sessionID = "session_id"
            case modelName = "model_name"
            case message
        }
    }

    private struct RemotePredownloadStatusResponse: Decodable {
        let success: Bool
        let sessionID: String
        let modelName: String?
        let status: String
        let progress: Int
        let queueStatus: RemoteQueueStatus?
        let error: String?

        enum CodingKeys: String, CodingKey {
            case success
            case sessionID = "session_id"
            case modelName = "model_name"
            case status
            case progress
            case queueStatus = "queue_status"
            case error
        }
    }

    private struct RemoteQueueStatus: Decodable {
        let message: String?
        let repoID: String?
        let stageName: String?
        let stageIndex: Int?
        let stageTotal: Int?
        let downloadPercent: Int?

        enum CodingKeys: String, CodingKey {
            case message
            case repoID = "repo_id"
            case stageName = "stage_name"
            case stageIndex = "stage_index"
            case stageTotal = "stage_total"
            case downloadPercent = "download_percent"
        }
    }

    private struct StableAudioCheckpointsResponse: Decodable {
        let success: Bool
        let repo: String?
        let checkpoints: [String]?
        let count: Int?
        let error: String?
    }

    private struct StableAudioPredownloadInventoryResponse: Decodable {
        let success: Bool
        let knownModels: [StableAudioKnownModelRow]
        let finetuneRepo: String?
        let finetuneCheckpoints: [StableAudioCheckpointInventoryRow]
        let cachedFinetunes: [String]
        let error: String?

        enum CodingKeys: String, CodingKey {
            case success
            case knownModels = "known_models"
            case finetuneRepo = "finetune_repo"
            case finetuneCheckpoints = "finetune_checkpoints"
            case cachedFinetunes = "cached_finetunes"
            case error
        }
    }

    private struct StableAudioKnownModelRow: Decodable {
        let repoID: String
        let label: String
        let downloaded: Bool
        let missing: [String]

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case label
            case downloaded
            case missing
        }
    }

    private struct StableAudioCheckpointInventoryRow: Decodable {
        let name: String
        let downloaded: Bool
    }

    private struct StableAudioModelSwitchResponse: Decodable {
        let success: Bool?
        let message: String?
        let error: String?
    }

    private struct SA3PredownloadInventoryResponse: Decodable {
        let success: Bool
        let knownModels: [SA3KnownModelRow]
        let error: String?

        enum CodingKeys: String, CodingKey {
            case success
            case knownModels = "known_models"
            case error
        }
    }

    private struct SA3KnownModelRow: Decodable {
        let repoID: String
        let label: String
        let downloaded: Bool
        let missing: [String]

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case label
            case downloaded
            case missing
        }
    }

    init() {
        let savedBackend = UserDefaults.standard.string(forKey: Self.stableAudioBackendDefaultsKey) ?? StableAudioBackendEngine.mps.rawValue
        stableAudioBackendEngine = StableAudioBackendEngine.from(rawValue: savedBackend)
        let savedMelodyFlowBackend = UserDefaults.standard.string(
            forKey: Self.melodyFlowBackendDefaultsKey
        ) ?? MelodyFlowBackendEngine.mps.rawValue
        melodyFlowBackendEngine = MelodyFlowBackendEngine.from(rawValue: savedMelodyFlowBackend)
        let savedCareyBackend = UserDefaults.standard.string(
            forKey: Self.careyBackendDefaultsKey
        ) ?? CareyBackendEngine.mlx.rawValue
        careyBackendEngine = CareyBackendEngine.from(rawValue: savedCareyBackend)
        careyUseXlModels = UserDefaults.standard.bool(forKey: Self.careyUseXlModelsDefaultsKey)
        careyUseScragVae = UserDefaults.standard.bool(forKey: Self.careyUseScragVaeDefaultsKey)
        if UserDefaults.standard.object(forKey: Self.careyUseSampledMlxVaeEncodeDefaultsKey) != nil {
            careyUseSampledMlxVaeEncode = UserDefaults.standard.bool(
                forKey: Self.careyUseSampledMlxVaeEncodeDefaultsKey
            )
        } else {
            careyUseSampledMlxVaeEncode = true
        }
        if let storedSA3Settings = storedSA3RuntimeSettings() {
            applySA3RuntimeSettingsToDraft(storedSA3Settings)
        }
        observeApplicationTermination()
        refreshStableAudioTokenState()
        loadManifest()
        observeSA3TrainingCompletion()
    }

    deinit {
        logRefreshTask?.cancel()
        modelDownloadPollTask?.cancel()
        garyLocalDownloadTask?.cancel()
        foundationLocalDownloadTask?.cancel()
        melodyflowLocalDownloadTask?.cancel()
        stableAudioLocalDownloadTask?.cancel()
        sa3LocalDownloadTask?.cancel()
        careyDownloadTask?.cancel()
    }

    var modelDownloadSections: [DownloadModelSection] {
        let grouped = Dictionary(grouping: downloadableModels, by: \.size)
        let order = ["small", "medium", "large"]
        return order.compactMap { size in
            guard let models = grouped[size], !models.isEmpty else { return nil }
            return DownloadModelSection(
                id: size,
                title: size.capitalized,
                models: models.sorted { lhs, rhs in
                    lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
                }
            )
        }
    }

    var canManageModelDownloads: Bool {
        canManageModelDownloads(for: modelDownloadServiceID)
    }

    func canManageModelDownloads(for serviceID: String) -> Bool {
        guard let runtime = manager?.services.first(where: { $0.id == serviceID }) else {
            return false
        }
        if runtime.isBootstrapping {
            return false
        }
        if runtime.processState == .running {
            return true
        }
        switch serviceID {
        case "audiocraft_mlx":
            return canRunGaryOfflineDownloads
        case "foundation":
            return canRunFoundationOfflineDownloads
        case "melodyflow":
            return canRunMelodyflowOfflineDownloads
        case "sa3":
            return canRunSA3OfflineDownloads
        case "stable_audio":
            return canRunStableAudioOfflineDownloads
        case "carey":
            return canRunCareyFocusedDownload
        default:
            return false
        }
    }

    var modelDownloadServiceDisplayName: String {
        modelDownloadDisplayName(forServiceID: modelDownloadServiceID)
    }

    var canManageStableAudioPredownloads: Bool {
        guard modelDownloadServiceID == "stable_audio" else { return false }
        return canManageModelDownloads && stableAudioTokenConfigured
    }

    var canManageSA3Predownloads: Bool {
        guard modelDownloadServiceID == "sa3" else { return false }
        return canManageModelDownloads && stableAudioTokenConfigured
    }

    var showsExperimentalCareyMlxVaeEncodeToggle: Bool {
        Self.isExperimentalCareyMlxVaeEncodeToggleEnabled()
    }

    var canRunCareyFocusedDownload: Bool {
        guard let runtime = manager?.services.first(where: { $0.id == "carey" }) else {
            return false
        }
        return resolveCareyDownloadScriptURL(for: runtime) != nil
    }

    var canRunGaryOfflineDownloads: Bool {
        guard let runtime = currentGaryRuntime() else {
            return false
        }
        guard FileManager.default.fileExists(atPath: runtime.service.executable.path) else {
            return false
        }
        return garyPredownloadHelperURL(for: runtime) != nil
    }

    var canRunFoundationOfflineDownloads: Bool {
        guard let runtime = currentFoundationRuntime() else {
            return false
        }
        guard FileManager.default.fileExists(atPath: runtime.service.executable.path) else {
            return false
        }
        return foundationPredownloadHelperURL(for: runtime) != nil
    }

    var canRunMelodyflowOfflineDownloads: Bool {
        guard let runtime = currentMelodyflowRuntime() else {
            return false
        }
        guard FileManager.default.fileExists(atPath: runtime.service.executable.path) else {
            return false
        }
        return melodyflowPredownloadHelperURL(for: runtime) != nil
    }

    var canRunSA3OfflineDownloads: Bool {
        guard let runtime = currentSA3Runtime() else {
            return false
        }
        guard FileManager.default.fileExists(atPath: runtime.service.executable.path) else {
            return false
        }
        return sa3PredownloadHelperURL(for: runtime) != nil
    }

    var canRunStableAudioOfflineDownloads: Bool {
        guard let runtime = currentStableAudioRuntime() else {
            return false
        }
        guard FileManager.default.fileExists(atPath: runtime.service.executable.path) else {
            return false
        }
        return stableAudioPredownloadHelperURL(for: runtime) != nil
    }

    var isCareyServiceRunning: Bool {
        manager?.services.first(where: { $0.id == "carey" })?.processState == .running
    }

    var isCareyEnvironmentReady: Bool {
        careyPythonExecutableURL() != nil
    }

    var isCareyScragVaeDownloaded: Bool {
        if !careyOptionalModels.isEmpty {
            return careyOptionalModels.allSatisfy(\.downloaded)
        }
        guard let runtime = manager?.services.first(where: { $0.id == "carey" }) else { return false }
        let checkpointDirectory = resolveCareyCheckpointDirectory(for: runtime)
        let fileManager = FileManager.default
        return Self.careyOptionalModelFiles.allSatisfy { row in
            let fileURL = Self.resolveCareyModelFileURL(
                baseCheckpointDirectory: checkpointDirectory,
                relativePath: row.relativePath
            )
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: fileURL.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue,
                  let attrs = try? fileManager.attributesOfItem(atPath: fileURL.path),
                  let size = attrs[.size] as? NSNumber else {
                return false
            }
            return size.int64Value > 0
        }
    }

    var canEnableCareyScragVae: Bool {
        isCareyScragVaeDownloaded || careyUseScragVae
    }

    var isSA3ServiceRunning: Bool {
        manager?.services.first(where: { $0.id == "sa3" })?.processState == .running
    }

    var isSA3EnvironmentReady: Bool {
        sa3PythonExecutableURL() != nil
    }

    var isSA3AutolabelActive: Bool {
        sa3AutolabelManager.state?.isActive == true
    }

    var careyLoraRequiresMpsBackend: Bool {
        careyBackendEngine == .mlx
    }

    private func activeCareyConfigNames() -> (base: String, sft: String, turbo: String, lego: String) {
        if careyUseXlModels {
            return ("acestep-v15-xl-base", "acestep-v15-xl-sft", "acestep-v15-xl-turbo", "acestep-v15-xl-base")
        }
        return ("acestep-v15-base", "acestep-v15-sft", "acestep-v15-turbo", "acestep-v15-base")
    }

    private func activeCareyRequiredModelFiles() -> [(label: String, relativePath: String)] {
        requiredCareyModelFiles(for: [.base, .sft, .turbo])
    }

    private func requiredCareyModelFiles(
        for targets: [CareyDownloadTarget]
    ) -> [(label: String, relativePath: String)] {
        let configs = activeCareyConfigNames()
        let targetSet = Set(targets)
        let includesShared = targetSet.contains(.shared)
            || targetSet.contains(.base)
            || targetSet.contains(.sft)
            || targetSet.contains(.turbo)

        var files: [(label: String, relativePath: String)] = []
        if targetSet.contains(.base) {
            files += Self.careyDiTRequiredFiles(labelPrefix: "DiT Base", configName: configs.base)
            if configs.lego != configs.base {
                files += Self.careyDiTRequiredFiles(labelPrefix: "DiT Lego Base", configName: configs.lego)
            }
        }
        if targetSet.contains(.sft) {
            files += Self.careyDiTRequiredFiles(labelPrefix: "DiT SFT", configName: configs.sft)
        }
        if targetSet.contains(.turbo) {
            files += Self.careyDiTRequiredFiles(labelPrefix: "DiT Turbo", configName: configs.turbo)
        }
        if includesShared {
            files += Self.careySharedRequiredModelFiles
        }
        if targetSet.contains(.scragVae) {
            files += Self.careyOptionalModelFiles
        }
        return files
    }

    private func currentCareyDownloadFiles() -> [(label: String, relativePath: String)] {
        if careyActiveDownloadTargets.isEmpty {
            return activeCareyRequiredModelFiles()
        }
        return requiredCareyModelFiles(for: careyActiveDownloadTargets)
    }

    func startCareyFocusedDownload(for target: CareyDownloadTarget) {
        startCareyFocusedDownload(targets: [target])
    }

    func isCareyDownloadTargetActive(_ target: CareyDownloadTarget) -> Bool {
        careyActiveDownloadTargets.contains(target)
    }

    func loadManifest() {
        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        garyLocalDownloadTask?.cancel()
        garyLocalDownloadTask = nil
        foundationLocalDownloadTask?.cancel()
        foundationLocalDownloadTask = nil
        melodyflowLocalDownloadTask?.cancel()
        melodyflowLocalDownloadTask = nil
        stableAudioLocalDownloadTask?.cancel()
        stableAudioLocalDownloadTask = nil
        sa3LocalDownloadTask?.cancel()
        sa3LocalDownloadTask = nil
        careyDownloadTask?.cancel()
        careyDownloadTask = nil
        isModelDownloadInProgress = false
        isCareyDownloadInProgress = false
        isCareyLifecycleActionInProgress = false
        activeModelDownloadPath = nil
        activeModelDownloadSessionID = nil
        careyActiveDownloadTargets = []
        careyRequiredModels = []
        careyOptionalModels = []
        rebuildFailureReport = nil
        rebuildDiagnosticsStatusMessage = ""
        managerCancellables.removeAll()

        let defaultURL = ManifestLoader.defaultManifestURL()
        manifestPath = defaultURL.path
        reloadHFScreenshots()
        refreshStableAudioTokenState()

        do {
            let manifest = try ManifestLoader.load(from: defaultURL)
            applySA3ManifestDefaults(from: manifest)
            let manager = ServiceManager(manifest: manifest)
            manager.setStableAudioBackendEngine(stableAudioBackendEngine.rawValue, restartIfRunning: false)
            manager.setMelodyFlowBackendEngine(melodyFlowBackendEngine.rawValue, restartIfRunning: false)
            manager.setCareyBackendEngine(careyBackendEngine.rawValue, restartIfRunning: false)
            manager.setCareyUseXlModels(careyUseXlModels, restartIfRunning: false)
            manager.setCareyUseScragVae(careyUseScragVae, restartIfRunning: false)
            manager.setCareyUseSampledMlxVaeEncode(careyUseSampledMlxVaeEncode, restartIfRunning: false)
            manager.setSA3RuntimeSettings(currentSA3RuntimeSettings().normalized, restartIfRunning: false)
            self.manager = manager
            bindManager(manager)
            startupError = nil
            selectedServiceID = manager.services.first?.id
            selectedLogText = ""
            isLogViewerPinnedToBottom = true
            lastLogMetadataByService.removeAll()
            manager.startAutoStartServices()
            startLogRefreshLoop()
            requestLogRefresh(force: true)
        } catch {
            self.manager = nil
            managerCancellables.removeAll()
            startupError = error.localizedDescription
            selectedLogText = ""
            lastLogMetadataByService.removeAll()
            logRefreshTask?.cancel()
            logRefreshTask = nil
            downloadableModels = []
            careyRequiredModels = []
            careyOptionalModels = []
            modelDownloadStatusMessage = ""
            isModelCatalogLoading = false
        }
    }

    func clearRebuildFailureReport() {
        manager?.clearLatestRebuildFailure()
        rebuildFailureReport = nil
        rebuildDiagnosticsStatusMessage = ""
        isRequirementsEditorPresented = false
        requirementsEditorStatusMessage = ""
    }

    func openRebuildFailureLogFile() {
        guard let report = rebuildFailureReport else { return }
        NSWorkspace.shared.open(report.logFile)
    }

    func openRebuildFailureRequirementsFile() {
        guard let report = rebuildFailureReport,
              let requirementsFile = resolvedRequirementsFile(for: report) else { return }
        NSWorkspace.shared.open(requirementsFile)
    }

    func copyRebuildFailureDiagnostics() {
        guard let report = rebuildFailureReport else { return }
        copyTextToPasteboard(diagnosticsReportText(for: report))
        rebuildDiagnosticsStatusMessage = "diagnostics copied."
    }

    func openSupportEmail() {
        guard let report = rebuildFailureReport else { return }
        let subject = "gary4local repair help (\(report.serviceID))"
        let diagnostics = diagnosticsReportText(for: report)
        let fullBody = """
        service: \(report.serviceID)
        summary: \(report.summary)

        diagnostics:

        \(diagnostics)
        """

        if let url = supportEmailURL(subject: subject, body: fullBody) {
            NSWorkspace.shared.open(url)
            return
        }

        let truncatedBody = String(fullBody.prefix(6000)) + "\n\n[diagnostics truncated in draft]"
        if let url = supportEmailURL(subject: subject, body: truncatedBody) {
            copyTextToPasteboard(diagnostics)
            rebuildDiagnosticsStatusMessage = "email draft opened with shortened diagnostics. full diagnostics copied."
            NSWorkspace.shared.open(url)
            return
        }

        copyTextToPasteboard(diagnostics)
        rebuildDiagnosticsStatusMessage = "could not open email draft. diagnostics copied."
    }

    func openSupportDiscord() {
        guard let url = URL(string: "https://discord.gg/xUkpsKNvM6") else { return }
        NSWorkspace.shared.open(url)
    }

    private func supportEmailURL(subject: String, body: String) -> URL? {
        var components = URLComponents()
        components.scheme = "mailto"
        components.path = "kev@thecollabagepatch.com"
        components.queryItems = [
            URLQueryItem(name: "subject", value: subject),
            URLQueryItem(name: "body", value: body)
        ]
        return components.url
    }

    private func copyTextToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    func retryRebuildFailure() {
        guard let report = rebuildFailureReport else { return }
        rebuildDiagnosticsStatusMessage = "running repair again..."
        manager?.rebuildEnvironment(serviceID: report.serviceID)
    }

    func cleanRepairRebuildFailure() {
        guard let report = rebuildFailureReport else { return }
        rebuildDiagnosticsStatusMessage = "starting repair from scratch..."
        manager?.rebuildEnvironment(
            serviceID: report.serviceID,
            forceRecreateVenv: true,
            extraPipArguments: ["--no-cache-dir"]
        )
    }

    func openRebuildFailureRequirementsEditor() {
        guard let report = rebuildFailureReport,
              let requirementsFile = resolvedRequirementsFile(for: report) else { return }
        do {
            requirementsEditorText = try String(contentsOf: requirementsFile, encoding: .utf8)
            requirementsEditorPath = requirementsFile.path
            requirementsEditorStatusMessage = ""
            isRequirementsEditorPresented = true
        } catch {
            rebuildDiagnosticsStatusMessage = "failed to open requirements: \(error.localizedDescription)"
        }
    }

    func saveRequirementsEditor() {
        guard !requirementsEditorPath.isEmpty else { return }
        do {
            try requirementsEditorText.write(
                toFile: requirementsEditorPath,
                atomically: true,
                encoding: .utf8
            )
            requirementsEditorStatusMessage = ""
            isRequirementsEditorPresented = false
            rebuildDiagnosticsStatusMessage = "requirements saved. run repair again."
        } catch {
            requirementsEditorStatusMessage = error.localizedDescription
        }
    }

    func closeRequirementsEditor() {
        isRequirementsEditorPresented = false
        requirementsEditorStatusMessage = ""
    }

    func selectService(_ serviceID: String) {
        guard selectedServiceID != serviceID else { return }
        selectedServiceID = serviceID
        selectedLogText = ""
        isLogViewerPinnedToBottom = true
        requestLogRefresh(force: true)
    }

    func refreshLog() {
        guard manager != nil, selectedServiceID != nil else {
            selectedLogText = ""
            return
        }
        requestLogRefresh(force: true)
    }

    func setStableAudioBackendEngine(_ backend: StableAudioBackendEngine) {
        guard stableAudioBackendEngine != backend else { return }
        stableAudioBackendEngine = backend
        UserDefaults.standard.set(backend.rawValue, forKey: Self.stableAudioBackendDefaultsKey)
        manager?.setStableAudioBackendEngine(backend.rawValue, restartIfRunning: true)
        if manager?.services.first(where: { $0.id == "stable_audio" })?.isRunning == true {
            stableAudioTokenStatus = "stable audio backend set to \(backend.displayName). service restarting..."
        } else {
            stableAudioTokenStatus = "stable audio backend set to \(backend.displayName)."
        }
    }

    func setMelodyFlowBackendEngine(_ backend: MelodyFlowBackendEngine) {
        guard melodyFlowBackendEngine != backend else { return }
        melodyFlowBackendEngine = backend
        UserDefaults.standard.set(backend.rawValue, forKey: Self.melodyFlowBackendDefaultsKey)
        manager?.setMelodyFlowBackendEngine(backend.rawValue, restartIfRunning: true)
        if manager?.services.first(where: { $0.id == "melodyflow" })?.isRunning == true {
            melodyFlowBackendStatus = "melodyflow backend set to \(backend.displayName). service restarting..."
        } else {
            melodyFlowBackendStatus = "melodyflow backend set to \(backend.displayName)."
        }
    }

    func setCareyBackendEngine(_ backend: CareyBackendEngine) {
        guard careyBackendEngine != backend else { return }
        careyBackendEngine = backend
        UserDefaults.standard.set(backend.rawValue, forKey: Self.careyBackendDefaultsKey)
        manager?.setCareyBackendEngine(backend.rawValue, restartIfRunning: true)
        if manager?.services.first(where: { $0.id == "carey" })?.isRunning == true {
            careyBackendStatus = "carey backend set to \(backend.displayName). service restarting..."
        } else {
            careyBackendStatus = "carey backend set to \(backend.displayName)."
        }
    }

    func setCareyUseXlModels(_ enabled: Bool) {
        guard careyUseXlModels != enabled else { return }
        careyUseXlModels = enabled
        UserDefaults.standard.set(enabled, forKey: Self.careyUseXlModelsDefaultsKey)
        manager?.setCareyUseXlModels(enabled, restartIfRunning: true)
        if modelDownloadServiceID == "carey" {
            prepareCareyPredownloadState()
        }
        if manager?.services.first(where: { $0.id == "carey" })?.isRunning == true {
            careyBackendStatus = enabled
                ? "carey xl models enabled. service restarting..."
                : "regular carey models enabled. service restarting..."
        } else {
            careyBackendStatus = enabled
                ? "carey xl models enabled for the next start."
                : "regular carey models enabled for the next start."
        }
    }

    func setCareyUseScragVae(_ enabled: Bool) {
        guard careyUseScragVae != enabled else { return }
        if enabled && !isCareyScragVaeDownloaded {
            careyBackendStatus = "download ScragVAE first, then this switch will wake up."
            return
        }

        careyUseScragVae = enabled
        UserDefaults.standard.set(enabled, forKey: Self.careyUseScragVaeDefaultsKey)
        manager?.setCareyUseScragVae(enabled, restartIfRunning: true)

        if manager?.services.first(where: { $0.id == "carey" })?.isRunning == true {
            careyBackendStatus = enabled
                ? "ScragVAE enabled. carey is restarting..."
                : "stock VAE enabled. carey is restarting..."
        } else {
            careyBackendStatus = enabled
                ? "ScragVAE enabled for the next carey start."
                : "stock VAE enabled for the next carey start."
        }
    }

    func setCareyUseSampledMlxVaeEncode(_ enabled: Bool) {
        guard careyUseSampledMlxVaeEncode != enabled else { return }
        careyUseSampledMlxVaeEncode = enabled
        UserDefaults.standard.set(enabled, forKey: Self.careyUseSampledMlxVaeEncodeDefaultsKey)
        manager?.setCareyUseSampledMlxVaeEncode(enabled, restartIfRunning: true)

        let usesMlxBackend = careyBackendEngine == .mlx
        if manager?.services.first(where: { $0.id == "carey" })?.isRunning == true {
            if usesMlxBackend {
                careyBackendStatus = enabled
                    ? "carey MLX VAE sampled encode enabled. service restarting..."
                    : "carey MLX VAE mean encode enabled. service restarting..."
            } else {
                careyBackendStatus = enabled
                    ? "carey sampled MLX VAE encode saved. it will apply the next time MLX backend is used."
                    : "carey mean MLX VAE encode saved. it will apply the next time MLX backend is used."
            }
        } else {
            if usesMlxBackend {
                careyBackendStatus = enabled
                    ? "carey MLX VAE sampled encode enabled for the next start."
                    : "carey MLX VAE mean encode enabled for the next start."
            } else {
                careyBackendStatus = enabled
                    ? "carey sampled MLX VAE encode saved for the next MLX start."
                    : "carey mean MLX VAE encode saved for the next MLX start."
            }
        }
    }

    func saveSA3RuntimeSettings() {
        let settings = currentSA3RuntimeSettings().normalized
        guard settings != storedSA3RuntimeSettings() else {
            sa3RuntimeSettingsStatus = "no sa3 advanced setting changes to save."
            return
        }

        storeSA3RuntimeSettings(settings)
        manager?.setSA3RuntimeSettings(settings, restartIfRunning: true)

        if manager?.services.first(where: { $0.id == "sa3" })?.isRunning == true {
            sa3RuntimeSettingsStatus = "sa3 advanced settings saved. service restarting..."
        } else {
            sa3RuntimeSettingsStatus = "sa3 advanced settings saved for the next start."
        }
    }

    func resetSA3RuntimeSettingsToDefaults() {
        applySA3RuntimeSettingsToDraft(sa3DefaultRuntimeSettings)
        sa3RuntimeSettingsStatus = "sa3 advanced settings reset to manifest defaults. save to apply."
    }

    private func currentSA3RuntimeSettings() -> SA3RuntimeSettings {
        SA3RuntimeSettings(
            peakNormalizeDb: sa3PeakNormalizeDb,
            limiterCeilingDb: sa3LimiterCeilingDb,
            latentRescale: sa3LatentRescale,
            latentShift: sa3LatentShift,
            latentTargetStd: sa3LatentTargetStd,
            continuationTailPad: sa3ContinuationTailPad,
            useFP32DiT: sa3UseFP32DiT
        )
    }

    private func applySA3RuntimeSettingsToDraft(_ settings: SA3RuntimeSettings) {
        sa3PeakNormalizeDb = settings.peakNormalizeDb
        sa3LimiterCeilingDb = settings.limiterCeilingDb
        sa3LatentRescale = settings.latentRescale
        sa3LatentShift = settings.latentShift
        sa3LatentTargetStd = settings.latentTargetStd
        sa3ContinuationTailPad = settings.continuationTailPad
        sa3UseFP32DiT = settings.useFP32DiT
    }

    private func applySA3ManifestDefaults(from manifest: ResolvedManifest) {
        let manifestSettings = manifest.services
            .first(where: { $0.id == "sa3" })
            .map { SA3RuntimeSettings(environment: $0.environment) }
            ?? .fallbackDefaults
        sa3DefaultRuntimeSettings = manifestSettings
        applySA3RuntimeSettingsToDraft(storedSA3RuntimeSettings() ?? manifestSettings)
    }

    private func storedSA3RuntimeSettings() -> SA3RuntimeSettings? {
        guard let data = UserDefaults.standard.data(forKey: Self.sa3RuntimeSettingsDefaultsKey) else {
            return nil
        }
        return try? JSONDecoder().decode(SA3RuntimeSettings.self, from: data)
    }

    private func storeSA3RuntimeSettings(_ settings: SA3RuntimeSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        UserDefaults.standard.set(data, forKey: Self.sa3RuntimeSettingsDefaultsKey)
    }

    private func observeSA3TrainingCompletion() {
        sa3TrainingManager.$state
            .compactMap { $0 }
            .filter { $0.status == "completed" && $0.jobId != nil }
            .sink { [weak self] state in
                guard let self, let jobID = state.jobId,
                      self.lastHandledSA3TrainingJobID != jobID else {
                    return
                }
                self.lastHandledSA3TrainingJobID = jobID
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    do {
                        self.sa3LoraState = try self.buildSA3LoraState()
                        let reloaded = try await self.reloadSA3AdaptersIfRunning()
                        if self.isSA3LoraSheetPresented {
                            self.sa3LoraStatusMessage = reloaded
                                ? "training completed; registry updated and sa3 reloaded."
                                : "training completed; registry updated."
                        }
                    } catch {
                        if self.isSA3LoraSheetPresented {
                            self.sa3LoraErrorMessage =
                                "training completed, but automatic LoRA registration "
                                + "failed: \(error.localizedDescription)"
                        }
                    }
                }
            }
            .store(in: &cancellables)
    }

    func openSA3LoraSheet() {
        isSA3LoraSheetPresented = true
        Task { await refreshSA3LoraState() }
    }

    func openSA3TrainingSheet() {
        sa3TrainingManager.refresh()
        isSA3TrainingSheetPresented = true
    }

    func openCareyAceTrainingSheet() {
        careyAceTrainingManager.refresh()
        isCareyAceTrainingSheetPresented = true
    }

    func startManagedService(_ serviceID: String) {
        guard !(serviceID == "carey" && isSA3AutolabelActive) else {
            sa3AutolabelManager.reportLaunchError(
                "cancel or finish SA3 auto-labeling before starting Carey."
            )
            return
        }
        manager?.start(serviceID: serviceID)
    }

    func startSA3LoraTraining(_ request: SA3LoraTrainingRequest) {
        guard let service = currentSA3Runtime()?.service else {
            sa3TrainingManager.clearError()
            return
        }
        sa3TrainingManager.start(
            request: request,
            service: service,
            huggingFaceToken: sharedHuggingFaceToken
        )
    }

    func startCareyAceTraining(_ request: CareyAceTrainingRequest) {
        guard sa3AutolabelManager.state?.isActive != true else {
            careyAceTrainingManager.reportLaunchError(
                "cancel or finish SA3 auto-labeling before starting Carey training."
            )
            return
        }
        guard let service = currentCareyRuntime()?.service,
              let pythonURL = careyPythonExecutableURL() else {
            careyAceTrainingManager.clearError()
            return
        }
        careyAceTrainingManager.start(
            request: request,
            service: service,
            pythonURL: pythonURL,
            checkpointDirectory: careyCheckpointDirectoryURL(),
            loraCatalogURL: careyLoraCatalogURL(),
            loraRegistryURL: careyLoraRegistryURL(),
            captionsURL: careyCaptionsURL(),
            huggingFaceToken: sharedHuggingFaceToken
        )
    }

    func suggestSA3TrackMetadata(
        audioPath: String
    ) async throws -> SA3MetadataSuggestion {
        guard let service = currentSA3Runtime()?.service,
              let pythonURL = sa3PythonExecutableURL() else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2301,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Build the SA3 environment before suggesting BPM/key."
                ]
            )
        }
        return try await SA3AudioMetadataAnalyzer.analyze(
            audioPath: audioPath,
            service: service,
            pythonURL: pythonURL
        )
    }

    func startSA3Autolabel(
        datasetPath: String,
        style: SA3PromptStyle
    ) {
        guard let service = currentCareyRuntime()?.service,
              let pythonURL = careyPythonExecutableURL() else {
            sa3AutolabelManager.reportLaunchError(
                "Build the Carey environment before auto-labeling."
            )
            return
        }
        sa3AutolabelManager.start(
            datasetPath: datasetPath,
            style: style,
            service: service,
            pythonURL: pythonURL,
            huggingFaceToken: sharedHuggingFaceToken,
            careyServiceIsRunning: isCareyServiceRunning,
            careyTrainingIsActive: careyAceTrainingManager.state?.isActive == true
        )
    }

    func refreshSA3LoraState() async {
        isSA3LoraLoading = true
        sa3LoraErrorMessage = ""
        do {
            sa3LoraState = try buildSA3LoraState()
        } catch {
            sa3LoraErrorMessage = error.localizedDescription
        }
        isSA3LoraLoading = false
    }

    func saveSA3Lora(
        name: String,
        checkpointPath: String,
        promptsPath: String?
    ) async {
        isSA3LoraSaving = true
        sa3LoraErrorMessage = ""
        sa3LoraStatusMessage = ""
        sa3LoraBuildOutput = ""

        do {
            let normalizedName = try sanitizeSA3LoraName(name)
            let checkpointFile = expandedFileURL(
                from: checkpointPath,
                relativeTo: sa3WorkingDirectoryURL()
            )
            guard Self.looksLikeSA3LoraCheckpoint(checkpointFile) else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2201,
                    userInfo: [NSLocalizedDescriptionKey: "\(checkpointFile.path) does not look like an SA3 LoRA checkpoint file."]
                )
            }

            let trimmedPromptsPath = promptsPath?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .nilIfEmpty
            let promptsDirectory = trimmedPromptsPath.map {
                expandedFileURL(from: $0, relativeTo: sa3WorkingDirectoryURL())
            }
            if let promptsDirectory, !FileManager.default.fileExists(atPath: promptsDirectory.path) {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2202,
                    userInfo: [NSLocalizedDescriptionKey: "\(promptsDirectory.path) is not a valid prompts/source folder."]
                )
            }

            let catalogURL = sa3LoraCatalogURL()
            var catalog = try readSA3LoraCatalog(at: catalogURL)
            let existingStrength = catalog[normalizedName]?.strength ?? 1.0
            catalog[normalizedName] = SA3LoraCatalogEntry(
                path: checkpointFile.path,
                promptsPath: promptsDirectory?.path,
                strength: existingStrength
            )
            try saveSA3LoraCatalog(catalog, to: catalogURL)

            sa3LoraState = try buildSA3LoraState()
            let reloaded = try await reloadSA3AdaptersIfRunning()
            sa3LoraStatusMessage = reloaded
                ? "saved \(normalizedName) and reloaded sa3."
                : "saved \(normalizedName)."
        } catch {
            sa3LoraErrorMessage = error.localizedDescription
        }

        isSA3LoraSaving = false
    }

    func activateSA3LoraCheckpoint(named name: String, step: Int) async {
        guard sa3LoraSwitchingName == nil else { return }
        sa3LoraSwitchingName = name
        sa3LoraErrorMessage = ""
        sa3LoraStatusMessage = ""
        sa3LoraBuildOutput = ""

        do {
            let normalizedName = try sanitizeSA3LoraName(name)
            let catalogURL = sa3LoraCatalogURL()
            let originalCatalog = try readSA3LoraCatalog(at: catalogURL)
            guard var entry = originalCatalog[normalizedName],
                  entry.trainingJobId != nil,
                  !entry.trainingCheckpoints.isEmpty else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2208,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "\(normalizedName) was not registered by Gary's SA3 trainer."
                    ]
                )
            }
            guard let checkpoint = entry.trainingCheckpoints.first(
                where: { $0.step == step }
            ) else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2209,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Training step \(step) is not registered for \(normalizedName)."
                    ]
                )
            }

            let sourceURL = expandedFileURL(
                from: checkpoint.path,
                relativeTo: sa3WorkingDirectoryURL()
            )
            guard Self.looksLikeSA3LoraCheckpoint(sourceURL) else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2210,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Training checkpoint is missing: \(sourceURL.path)"
                    ]
                )
            }

            let destinationURL = sa3LoraDirectoryURL()
                .appendingPathComponent("\(normalizedName).safetensors")
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let nonce = UUID().uuidString.lowercased()
            let stagedURL = destinationURL.deletingLastPathComponent()
                .appendingPathComponent(".\(destinationURL.lastPathComponent).\(nonce).tmp")
            let backupURL = destinationURL.deletingLastPathComponent()
                .appendingPathComponent(".\(destinationURL.lastPathComponent).\(nonce).bak")
            let destinationExisted = FileManager.default.fileExists(
                atPath: destinationURL.path
            )

            try FileManager.default.copyItem(at: sourceURL, to: stagedURL)
            let sourceSize = try sourceURL.resourceValues(
                forKeys: [.fileSizeKey]
            ).fileSize
            let stagedSize = try stagedURL.resourceValues(
                forKeys: [.fileSizeKey]
            ).fileSize
            guard sourceSize == stagedSize else {
                try? FileManager.default.removeItem(at: stagedURL)
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2211,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Checkpoint staging was incomplete for \(normalizedName)."
                    ]
                )
            }

            if destinationExisted {
                try Self.renameReplacingNothing(
                    source: destinationURL,
                    destination: backupURL
                )
            }
            do {
                try Self.renameReplacingNothing(
                    source: stagedURL,
                    destination: destinationURL
                )
            } catch {
                if destinationExisted {
                    try? Self.renameReplacingNothing(
                        source: backupURL,
                        destination: destinationURL
                    )
                }
                try? FileManager.default.removeItem(at: stagedURL)
                throw error
            }

            do {
                var updatedCatalog = originalCatalog
                entry.path = destinationURL.path
                entry.selectedTrainingStep = step
                updatedCatalog[normalizedName] = entry
                try saveSA3LoraCatalog(updatedCatalog, to: catalogURL)
                sa3LoraState = try buildSA3LoraState()
            } catch {
                try? FileManager.default.removeItem(at: destinationURL)
                if destinationExisted {
                    try? Self.renameReplacingNothing(
                        source: backupURL,
                        destination: destinationURL
                    )
                }
                try? saveSA3LoraCatalog(originalCatalog, to: catalogURL)
                sa3LoraState = try? buildSA3LoraState()
                throw error
            }

            try? FileManager.default.removeItem(at: backupURL)

            do {
                let reloaded = try await reloadSA3AdaptersIfRunning()
                sa3LoraStatusMessage = reloaded
                    ? "\(normalizedName) now uses training step \(step); sa3 reloaded."
                    : "\(normalizedName) now uses training step \(step)."
            } catch {
                sa3LoraErrorMessage =
                    "\(normalizedName) now uses training step \(step) on disk, "
                    + "but sa3 could not reload it: \(error.localizedDescription)"
            }
        } catch {
            sa3LoraErrorMessage = error.localizedDescription
        }

        sa3LoraSwitchingName = nil
    }

    func removeSA3Lora(named name: String) async {
        sa3LoraErrorMessage = ""
        sa3LoraStatusMessage = ""
        sa3LoraBuildOutput = ""

        do {
            let normalizedName = try sanitizeSA3LoraName(name)
            let catalogURL = sa3LoraCatalogURL()
            var catalog = try readSA3LoraCatalog(at: catalogURL)
            catalog.removeValue(forKey: normalizedName)
            try saveSA3LoraCatalog(catalog, to: catalogURL)
            sa3LoraState = try buildSA3LoraState()
            let reloaded = try await reloadSA3AdaptersIfRunning()
            sa3LoraStatusMessage = reloaded
                ? "removed \(normalizedName) and reloaded sa3."
                : "removed \(normalizedName)."
        } catch {
            sa3LoraErrorMessage = error.localizedDescription
        }
    }

    func buildSA3LoraPrompts() async {
        isSA3LoraBuilding = true
        sa3LoraErrorMessage = ""
        sa3LoraStatusMessage = ""
        sa3LoraBuildOutput = ""

        do {
            let initialState = try buildSA3LoraState()
            guard let pythonURL = sa3PythonExecutableURL() else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2203,
                    userInfo: [NSLocalizedDescriptionKey: "SA3 must be built before prompts can be generated."]
                )
            }

            let scriptURL = sa3WorkingDirectoryURL().appendingPathComponent("build_lora_prompts.py")
            guard FileManager.default.fileExists(atPath: scriptURL.path) else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2204,
                    userInfo: [NSLocalizedDescriptionKey: "Missing \(scriptURL.path)"]
                )
            }

            let promptsDirectory = sa3PromptsDirectoryURL()
            try FileManager.default.createDirectory(
                at: promptsDirectory,
                withIntermediateDirectories: true,
                attributes: nil
            )

            var outputs: [String] = []
            for entry in initialState.entries where entry.registered && entry.captionCount > 0 {
                guard let sourcePath = entry.resolvedPromptsPath else { continue }
                let result = Self.runLocalProcess(
                    executableURL: pythonURL,
                    arguments: [
                        scriptURL.path,
                        "--name", entry.name,
                        "--captions-dir", sourcePath,
                        "--out-dir", promptsDirectory.path,
                        "--force",
                    ],
                    currentDirectory: sa3WorkingDirectoryURL()
                )
                let combinedOutput = result.output.isEmpty ? "\(entry.name): no output" : result.output
                guard result.exitCode == 0 else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 2205,
                        userInfo: [NSLocalizedDescriptionKey: combinedOutput]
                    )
                }
                outputs.append(combinedOutput)
            }

            if outputs.isEmpty {
                outputs.append("No registered SA3 LoRAs with txt sidecars were found.")
            }

            try ensureDefaultSA3Prompts()
            sa3LoraState = try buildSA3LoraState()
            sa3LoraBuildOutput = outputs.joined(separator: "\n")
            sa3LoraStatusMessage = "prompt JSONs updated"
        } catch {
            sa3LoraErrorMessage = error.localizedDescription
        }

        isSA3LoraBuilding = false
    }

    func openCareyLoraSheet() {
        isCareyLoraSheetPresented = true
        Task { await refreshCareyLoraState() }
    }

    func refreshCareyLoraState() async {
        isCareyLoraLoading = true
        careyLoraErrorMessage = ""
        do {
            careyLoraState = try buildCareyLoraState()
        } catch {
            careyLoraErrorMessage = error.localizedDescription
        }
        isCareyLoraLoading = false
    }

    func saveCareyLora(
        name: String,
        checkpointPath: String,
        captionsPath: String?,
        modelFamily: CareyLoraModelFamily
    ) async {
        isCareyLoraSaving = true
        careyLoraErrorMessage = ""
        careyLoraStatusMessage = ""
        careyLoraBuildOutput = ""

        do {
            let normalizedName = try sanitizeCareyLoraName(name)
            let checkpointDirectory = expandedFileURL(from: checkpointPath)
            guard Self.looksLikeCareyLoraCheckpointDirectory(checkpointDirectory) else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2001,
                    userInfo: [NSLocalizedDescriptionKey: "\(checkpointDirectory.path) does not look like a LoRA checkpoint folder."]
                )
            }

            let trimmedCaptionsPath = captionsPath?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .nilIfEmpty
            let captionsDirectory = trimmedCaptionsPath.map(expandedFileURL(from:))
            if let captionsDirectory, !FileManager.default.fileExists(atPath: captionsDirectory.path) {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2002,
                    userInfo: [NSLocalizedDescriptionKey: "\(captionsDirectory.path) is not a valid captions/source folder."]
                )
            }

            let catalogURL = careyLoraCatalogURL()
            var catalog = try readCareyLoraCatalog(at: catalogURL)
            let existing = catalog[normalizedName]
            let (scale, backends, _) = try loadCareyLoraMetadata(
                checkpointDirectory: checkpointDirectory,
                captionsDirectory: captionsDirectory,
                existing: existing
            )

            catalog[normalizedName] = CareyLoraCatalogEntry(
                path: checkpointDirectory.path,
                captionsPath: captionsDirectory?.path,
                scale: scale,
                backends: backends,
                modelFamily: modelFamily.rawValue
            )
            try saveCareyLoraCatalog(catalog, to: catalogURL)

            careyLoraState = try buildCareyLoraState()
            let reloaded = await tryReloadCareyAdminIfRunning()
            careyLoraStatusMessage = reloaded
                ? "saved \(normalizedName) and reloaded carey."
                : "saved \(normalizedName)."
        } catch {
            careyLoraErrorMessage = error.localizedDescription
        }

        isCareyLoraSaving = false
    }

    func removeCareyLora(named name: String) async {
        careyLoraErrorMessage = ""
        careyLoraStatusMessage = ""
        careyLoraBuildOutput = ""

        do {
            let normalizedName = try sanitizeCareyLoraName(name)
            let catalogURL = careyLoraCatalogURL()
            var catalog = try readCareyLoraCatalog(at: catalogURL)
            catalog.removeValue(forKey: normalizedName)
            try saveCareyLoraCatalog(catalog, to: catalogURL)
            careyLoraState = try buildCareyLoraState()
            let reloaded = await tryReloadCareyAdminIfRunning()
            careyLoraStatusMessage = reloaded
                ? "removed \(normalizedName) and reloaded carey."
                : "removed \(normalizedName)."
        } catch {
            careyLoraErrorMessage = error.localizedDescription
        }
    }

    func buildCareyLoraCaptions() async {
        isCareyLoraBuilding = true
        careyLoraErrorMessage = ""
        careyLoraStatusMessage = ""
        careyLoraBuildOutput = ""

        do {
            let initialState = try buildCareyLoraState()
            guard let pythonURL = careyPythonExecutableURL() else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2003,
                    userInfo: [NSLocalizedDescriptionKey: "Carey must be built before captions can be generated."]
                )
            }
            let scriptURL = careyWrapperDirectoryURL().appendingPathComponent("build_captions.py")
            guard FileManager.default.fileExists(atPath: scriptURL.path) else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2004,
                    userInfo: [NSLocalizedDescriptionKey: "Missing \(scriptURL.path)."]
                )
            }

            let captionsURL = careyCaptionsURL()
            try FileManager.default.createDirectory(
                at: captionsURL.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: nil
            )

            var arguments = [scriptURL.path]
            for entry in initialState.entries where entry.registered && entry.captionCount > 0 {
                guard let resolvedCaptionsPath = entry.resolvedCaptionsPath else { continue }
                arguments.append("--lora")
                arguments.append("\(entry.name):\(resolvedCaptionsPath)")
            }
            arguments.append("-o")
            arguments.append(captionsURL.path)

            let result = Self.runLocalProcess(
                executableURL: pythonURL,
                arguments: arguments,
                currentDirectory: careyWrapperDirectoryURL()
            )

            careyLoraBuildOutput = result.output
            if result.exitCode != 0 {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: Int(result.exitCode),
                    userInfo: [NSLocalizedDescriptionKey: result.output.nilIfEmpty ?? "build_captions.py failed."]
                )
            }

            try ensureDefaultCareyCaptions()
            careyLoraState = try buildCareyLoraState()
            let reloaded = await tryReloadCareyAdminIfRunning()
            careyLoraStatusMessage = reloaded
                ? "captions.json updated and reloaded carey."
                : "captions.json updated."
        } catch {
            careyLoraErrorMessage = error.localizedDescription
        }

        isCareyLoraBuilding = false
    }

    func updateLogViewerPinnedToBottom(_ pinnedToBottom: Bool) {
        guard isLogViewerPinnedToBottom != pinnedToBottom else { return }
        isLogViewerPinnedToBottom = pinnedToBottom
        if pinnedToBottom {
            requestLogRefresh(force: true)
        }
    }

    func openModelDownloadSheet(for serviceID: String) {
        if isModelDownloadInProgress, modelDownloadServiceID != serviceID {
            modelDownloadStatusMessage = "a model download is already in progress for \(modelDownloadServiceDisplayName)."
            isModelDownloadSheetPresented = true
            return
        }
        let changedService = modelDownloadServiceID != serviceID
        modelDownloadServiceID = serviceID
        if changedService {
            downloadableModels = []
            isModelCatalogLoading = true
            modelDownloadStatusMessage = "loading model catalog..."
        }
        isModelDownloadSheetPresented = true
        if serviceID == "stable_audio" {
            prepareStableAudioPredownloadState()
        } else if serviceID == "sa3" {
            prepareSA3PredownloadState()
        } else if serviceID == "carey" {
            prepareCareyPredownloadState()
        } else {
            refreshModelCatalogAndStatuses()
        }
    }

    func refreshModelCatalogAndStatuses() {
        if modelDownloadServiceID == "stable_audio" {
            modelDownloadStatusMessage = "stable audio uses repo/checkpoint pre-download controls below."
            isModelCatalogLoading = false
            downloadableModels = []
            return
        }

        if modelDownloadServiceID == "sa3" {
            prepareSA3PredownloadState()
            return
        }

        if modelDownloadServiceID == "carey" {
            prepareCareyPredownloadState()
            return
        }

        if modelDownloadServiceID == "audiocraft_mlx",
           modelDownloadAPIBaseURL(for: modelDownloadServiceID) == nil,
           canRunGaryOfflineDownloads {
            refreshGaryOfflineModelCatalogAndStatuses()
            return
        }

        if modelDownloadServiceID == "foundation",
           modelDownloadAPIBaseURL(for: modelDownloadServiceID) == nil,
           canRunFoundationOfflineDownloads {
            refreshFoundationOfflineModelCatalogAndStatuses()
            return
        }

        if modelDownloadServiceID == "melodyflow",
           modelDownloadAPIBaseURL(for: modelDownloadServiceID) == nil,
           canRunMelodyflowOfflineDownloads {
            refreshMelodyflowOfflineModelCatalogAndStatuses()
            return
        }

        if isModelDownloadInProgress {
            modelDownloadStatusMessage = "a model download is already in progress."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        garyLocalDownloadTask?.cancel()
        garyLocalDownloadTask = nil
        foundationLocalDownloadTask?.cancel()
        foundationLocalDownloadTask = nil
        melodyflowLocalDownloadTask?.cancel()
        melodyflowLocalDownloadTask = nil
        stableAudioLocalDownloadTask?.cancel()
        stableAudioLocalDownloadTask = nil
        activeModelDownloadSessionID = nil
        activeModelDownloadPath = nil
        isModelDownloadInProgress = false

        guard let baseURL = modelDownloadAPIBaseURL(for: modelDownloadServiceID) else {
            downloadableModels = []
            modelDownloadStatusMessage = {
                switch modelDownloadServiceID {
                case "audiocraft_mlx":
                    return "build gary before managing model downloads."
                case "foundation":
                    return "build foundation-1 before managing model downloads."
                case "melodyflow":
                    return "build terry (melodyflow) before managing model downloads."
                default:
                    return "start \(modelDownloadServiceDisplayName) to manage model downloads."
                }
            }()
            isModelCatalogLoading = false
            return
        }

        isModelCatalogLoading = true
        modelDownloadStatusMessage = "loading model catalog..."

        Task { [weak self] in
            guard let self else { return }
            do {
                let decoder = JSONDecoder()

                let catalogURL = baseURL.appendingPathComponent("api/models")
                let (catalogData, catalogResponse) = try await URLSession.shared.data(from: catalogURL)
                try self.ensureHTTP200(response: catalogResponse, body: catalogData)
                let remoteCatalog = try decoder.decode(RemoteModelsResponse.self, from: catalogData)
                guard remoteCatalog.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "failed to load model catalog."]
                    )
                }

                let statusURL = baseURL.appendingPathComponent("api/models/download_status")
                let (statusData, statusResponse) = try await URLSession.shared.data(from: statusURL)
                try self.ensureHTTP200(response: statusResponse, body: statusData)
                let remoteStatuses = try decoder.decode(RemoteDownloadStatusResponse.self, from: statusData)
                guard remoteStatuses.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "failed to load download statuses."]
                    )
                }

                var models = self.flattenRemoteModels(remoteCatalog.models)
                for index in models.indices {
                    if let status = remoteStatuses.models[models[index].path] {
                        models[index].downloaded = status.downloaded
                        if status.downloaded {
                            models[index].statusMessage = "downloaded"
                        } else if let missing = status.missing, !missing.isEmpty {
                            models[index].statusMessage = "missing \(missing.count) dependency\(missing.count == 1 ? "" : "ies")"
                        } else {
                            models[index].statusMessage = "not downloaded"
                        }
                    } else {
                        models[index].statusMessage = "unknown"
                    }
                }

                self.downloadableModels = models
                self.modelDownloadStatusMessage = "pick a model to pre-download for offline usage."
            } catch {
                self.downloadableModels = []
                self.modelDownloadStatusMessage = error.localizedDescription
            }
            self.isModelCatalogLoading = false
        }
    }

    func fetchStableAudioPredownloadCheckpoints() {
        let serviceID = modelDownloadServiceID
        guard serviceID == "stable_audio" else { return }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            if canRunStableAudioOfflineDownloads {
                fetchStableAudioOfflinePredownloadCheckpoints()
                return
            }
            modelDownloadStatusMessage = "build jerry before fetching checkpoints."
            return
        }

        let repo = stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty else {
            modelDownloadStatusMessage = "enter a hugging face repo first."
            return
        }
        guard !isStableAudioCheckpointFetchInProgress else { return }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        isStableAudioCheckpointFetchInProgress = true
        modelDownloadStatusMessage = "fetching checkpoints from \(repo)..."

        Task { [weak self] in
            guard let self else { return }
            defer { self.isStableAudioCheckpointFetchInProgress = false }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("models/checkpoints"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: ["finetune_repo": repo])

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let payload = try JSONDecoder().decode(StableAudioCheckpointsResponse.self, from: data)
                guard payload.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 6,
                        userInfo: [
                            NSLocalizedDescriptionKey: payload.error ?? "failed to fetch checkpoints."
                        ]
                    )
                }

                let checkpoints = (payload.checkpoints ?? []).sorted {
                    $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
                }
                self.stableAudioPredownloadCheckpoints = checkpoints
                self.stableAudioPredownloadSelectedCheckpoint = checkpoints.first ?? ""
                self.stableAudioPredownloadCheckpointDownloaded = [:]
                if checkpoints.isEmpty {
                    self.modelDownloadStatusMessage = "no .ckpt files found in \(repo)."
                } else {
                    self.modelDownloadStatusMessage = "\(checkpoints.count) checkpoint\(checkpoints.count == 1 ? "" : "s") found."
                }
                self.refreshStableAudioPredownloadInventory(checkpointsHint: checkpoints)
            } catch {
                self.stableAudioPredownloadCheckpoints = []
                self.stableAudioPredownloadCheckpointDownloaded = [:]
                self.stableAudioPredownloadSelectedCheckpoint = ""
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    func startStableAudioPredownloadOpenOne() {
        startStableAudioPredownload(
            payload: [
                "target_type": "pretrained",
                "repo_id": "stabilityai/stable-audio-open-1.0",
                "require_token": true
            ],
            targetLabel: "stabilityai/stable-audio-open-1.0"
        )
    }

    func startStableAudioPredownloadSelectedCheckpoint() {
        let repo = stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let checkpoint = stableAudioPredownloadSelectedCheckpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty else {
            modelDownloadStatusMessage = "enter a finetune repo first."
            return
        }
        guard !checkpoint.isEmpty else {
            modelDownloadStatusMessage = "fetch and choose a checkpoint first."
            return
        }
        startStableAudioPredownload(
            payload: [
                "target_type": "finetune",
                "finetune_repo": repo,
                "finetune_checkpoint": checkpoint,
                "base_repo": "stabilityai/stable-audio-open-small",
                "require_token": false
            ],
            targetLabel: "\(repo)/\(checkpoint)"
        )
    }

    func useStableAudioSelectedCheckpoint() {
        let serviceID = modelDownloadServiceID
        guard serviceID == "stable_audio" else { return }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            modelDownloadStatusMessage = "start \(modelDownloadServiceDisplayName) to use a model."
            return
        }
        let repo = stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let checkpoint = stableAudioPredownloadSelectedCheckpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty else {
            modelDownloadStatusMessage = "enter a finetune repo first."
            return
        }
        guard !checkpoint.isEmpty else {
            modelDownloadStatusMessage = "fetch and choose a checkpoint first."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "wait for the current download to finish."
            return
        }
        guard !isStableAudioModelSwitchInProgress else {
            modelDownloadStatusMessage = "model switch already in progress."
            return
        }

        isStableAudioModelSwitchInProgress = true
        modelDownloadStatusMessage = "loading \(repo)/\(checkpoint) into jerry cache..."

        Task { [weak self] in
            guard let self else { return }
            defer { self.isStableAudioModelSwitchInProgress = false }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("models/switch"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: [
                    "model_type": "finetune",
                    "finetune_repo": repo,
                    "finetune_checkpoint": checkpoint
                ])

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let payload = try JSONDecoder().decode(StableAudioModelSwitchResponse.self, from: data)
                if payload.success == false {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 9,
                        userInfo: [NSLocalizedDescriptionKey: payload.error ?? payload.message ?? "failed to switch model."]
                    )
                }
                self.modelDownloadStatusMessage = payload.message ?? "selected model is now active in jerry."
            } catch {
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    private func startStableAudioPredownload(payload: [String: Any], targetLabel: String) {
        let serviceID = modelDownloadServiceID
        guard serviceID == "stable_audio" else { return }
        guard stableAudioTokenConfigured else {
            modelDownloadStatusMessage = "save your hugging face token first."
            return
        }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            if canRunStableAudioOfflineDownloads {
                startStableAudioOfflinePredownload(payload: payload, targetLabel: targetLabel)
                return
            }
            modelDownloadStatusMessage = "build jerry before pre-downloading models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        activeModelDownloadSessionID = nil
        activeModelDownloadPath = targetLabel
        stableAudioPredownloadTargetLabel = targetLabel
        stableAudioPredownloadProgress = 0
        isModelDownloadInProgress = true
        modelDownloadStatusMessage = "starting \(targetLabel)..."

        Task { [weak self] in
            guard let self else { return }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("models/predownload"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: payload)

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let startResponse = try JSONDecoder().decode(RemotePredownloadStartResponse.self, from: data)
                guard startResponse.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 7,
                        userInfo: [NSLocalizedDescriptionKey: "unable to start stable audio predownload."]
                    )
                }

                self.activeModelDownloadSessionID = startResponse.sessionID
                self.modelDownloadStatusMessage = startResponse.message ?? "downloading \(targetLabel)..."
                self.startModelDownloadPolling(
                    sessionID: startResponse.sessionID,
                    modelPath: targetLabel,
                    serviceID: serviceID,
                    baseURL: baseURL,
                    statusPathPrefix: "models/predownload_status"
                )
            } catch {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                self.stableAudioPredownloadProgress = 0
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    private func prepareStableAudioPredownloadState() {
        downloadableModels = []
        isModelCatalogLoading = false
        if !isModelDownloadInProgress {
            stableAudioPredownloadProgress = 0
            stableAudioPredownloadTargetLabel = ""
        }
        stableAudioInventoryModels = []
        stableAudioCachedFinetunes = []
        if stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            stableAudioPredownloadRepoInput = "thepatch/jerry_grunge"
        }
        refreshStableAudioPredownloadInventory(checkpointsHint: stableAudioPredownloadCheckpoints)
        if isModelDownloadInProgress, !stableAudioPredownloadTargetLabel.isEmpty {
            modelDownloadStatusMessage = "downloading \(stableAudioPredownloadTargetLabel)..."
            return
        }
        if stableAudioTokenConfigured {
            modelDownloadStatusMessage = "choose a stable model or fetch finetune checkpoints to pre-download."
        } else {
            modelDownloadStatusMessage = "save your hugging face token in jerry setup first."
        }
    }

    func startSA3PredownloadRequiredModels() {
        let serviceID = modelDownloadServiceID
        guard serviceID == "sa3" else { return }
        guard stableAudioTokenConfigured else {
            modelDownloadStatusMessage = "save your hugging face token first."
            return
        }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            if canRunSA3OfflineDownloads {
                startSA3OfflinePredownloadRequiredModels()
                return
            }
            modelDownloadStatusMessage = "start \(modelDownloadServiceDisplayName) to pre-download models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        activeModelDownloadSessionID = nil
        activeModelDownloadPath = "required sa3 models"
        sa3PredownloadTargetLabel = "required sa3 models"
        sa3PredownloadProgress = 0
        isModelDownloadInProgress = true
        modelDownloadStatusMessage = "starting required sa3 models..."

        Task { [weak self] in
            guard let self else { return }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("models/predownload"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: [
                    "target_type": "required"
                ])

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let startResponse = try JSONDecoder().decode(RemotePredownloadStartResponse.self, from: data)
                guard startResponse.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 10,
                        userInfo: [NSLocalizedDescriptionKey: "unable to start sa3 predownload."]
                    )
                }

                self.activeModelDownloadSessionID = startResponse.sessionID
                self.modelDownloadStatusMessage = startResponse.message ?? "downloading required sa3 models..."
                self.startModelDownloadPolling(
                    sessionID: startResponse.sessionID,
                    modelPath: "required sa3 models",
                    serviceID: serviceID,
                    baseURL: baseURL,
                    statusPathPrefix: "models/predownload_status"
                )
            } catch {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                self.sa3PredownloadProgress = 0
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    private func prepareSA3PredownloadState() {
        downloadableModels = []
        isModelCatalogLoading = false
        if !isModelDownloadInProgress {
            sa3PredownloadProgress = 0
            sa3PredownloadTargetLabel = ""
        }
        refreshSA3PredownloadInventory()
        if isModelDownloadInProgress, !sa3PredownloadTargetLabel.isEmpty {
            modelDownloadStatusMessage = "downloading \(sa3PredownloadTargetLabel)..."
            return
        }
        if stableAudioTokenConfigured {
            modelDownloadStatusMessage = (
                "download the required sa3 model files for offline use, or let the first /load fetch them on demand."
            )
        } else {
            modelDownloadStatusMessage = "save your hugging face token in the sa3 or jerry setup panel first."
        }
    }

    func refreshSA3PredownloadInventory() {
        let serviceID = modelDownloadServiceID
        guard serviceID == "sa3" else { return }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            if canRunSA3OfflineDownloads {
                refreshSA3OfflinePredownloadInventory()
            } else {
                sa3InventoryModels = []
            }
            return
        }

        Task { [weak self] in
            guard let self else { return }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("models/predownload_inventory"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: [:])

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let payload = try JSONDecoder().decode(SA3PredownloadInventoryResponse.self, from: data)
                guard payload.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 11,
                        userInfo: [NSLocalizedDescriptionKey: payload.error ?? "failed to load sa3 inventory."]
                    )
                }

                self.sa3InventoryModels = payload.knownModels.map { row in
                    SA3InventoryModelStatus(
                        repoID: row.repoID,
                        label: row.label,
                        downloaded: row.downloaded,
                        missing: row.missing
                    )
                }
            } catch {
                if self.sa3InventoryModels.isEmpty {
                    self.modelDownloadStatusMessage = error.localizedDescription
                }
            }
        }
    }

    func refreshStableAudioPredownloadInventory(checkpointsHint: [String] = []) {
        let serviceID = modelDownloadServiceID
        guard serviceID == "stable_audio" else { return }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            if canRunStableAudioOfflineDownloads {
                refreshStableAudioOfflinePredownloadInventory(checkpointsHint: checkpointsHint)
            } else {
                stableAudioInventoryModels = []
                stableAudioCachedFinetunes = []
                stableAudioPredownloadCheckpointDownloaded = [:]
            }
            return
        }

        let repo = stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let hint = checkpointsHint.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        Task { [weak self] in
            guard let self else { return }
            do {
                var body: [String: Any] = [:]
                if !repo.isEmpty {
                    body["finetune_repo"] = repo
                }
                if !hint.isEmpty {
                    body["checkpoints"] = hint
                }

                var request = URLRequest(url: baseURL.appendingPathComponent("models/predownload_inventory"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let payload = try JSONDecoder().decode(StableAudioPredownloadInventoryResponse.self, from: data)
                guard payload.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 8,
                        userInfo: [NSLocalizedDescriptionKey: payload.error ?? "failed to load stable inventory."]
                    )
                }

                self.stableAudioInventoryModels = payload.knownModels.map { row in
                    StableAudioInventoryModelStatus(
                        id: row.repoID,
                        label: row.label,
                        downloaded: row.downloaded,
                        missing: row.missing
                    )
                }
                self.stableAudioCachedFinetunes = payload.cachedFinetunes
                self.stableAudioPredownloadCheckpointDownloaded = Dictionary(
                    uniqueKeysWithValues: payload.finetuneCheckpoints.map { ($0.name, $0.downloaded) }
                )
            } catch {
                if self.stableAudioInventoryModels.isEmpty {
                    self.modelDownloadStatusMessage = error.localizedDescription
                }
            }
        }
    }

    private func prepareCareyPredownloadState() {
        downloadableModels = []
        isModelCatalogLoading = false
        if !isCareyDownloadInProgress {
            careyActiveDownloadTargets = []
            careyPredownloadProgress = 0
            careyPredownloadActiveLabel = ""
            careyProgressByLabel = [:]
        }
        refreshCareyPredownloadInventory()
        if isCareyDownloadInProgress {
            modelDownloadStatusMessage = "downloading carey model files..."
        } else if canRunCareyFocusedDownload {
            let familyDescription = careyUseXlModels ? "xl" : "regular"
            modelDownloadStatusMessage = "download the \(familyDescription) carey model family you want to use, or fetch all three at once. ScragVAE is optional."
        } else {
            modelDownloadStatusMessage = "focused download script not found (expected in runtime/scripts or workspace/scripts)."
        }
    }

    func refreshCareyPredownloadInventory() {
        guard modelDownloadServiceID == "carey" else { return }
        guard let runtime = manager?.services.first(where: { $0.id == "carey" }) else {
            careyRequiredModels = []
            careyOptionalModels = []
            return
        }

        let fileManager = FileManager.default
        let checkpointDirectory = resolveCareyCheckpointDirectory(for: runtime)
        let requiredModelFiles = activeCareyRequiredModelFiles()
        let inventoryRows = { (rows: [(label: String, relativePath: String)]) in
            rows.map { row in
                let fileURL = Self.resolveCareyModelFileURL(
                    baseCheckpointDirectory: checkpointDirectory,
                    relativePath: row.relativePath
                )
                var isDirectory: ObjCBool = false
                let exists = fileManager.fileExists(atPath: fileURL.path, isDirectory: &isDirectory) && !isDirectory.boolValue
                let bytes: Int64
                if exists,
                   let attrs = try? fileManager.attributesOfItem(atPath: fileURL.path),
                   let size = attrs[.size] as? NSNumber {
                    bytes = size.int64Value
                } else {
                    bytes = 0
                }
                return CareyRequiredModelStatus(
                    id: row.relativePath,
                    label: row.label,
                    relativePath: row.relativePath,
                    downloaded: exists && bytes > 0,
                    sizeBytes: bytes
                )
            }
        }
        let rows = inventoryRows(requiredModelFiles)
        let optionalRows = inventoryRows(Self.careyOptionalModelFiles)

        careyRequiredModels = rows
        careyOptionalModels = optionalRows
        if !isCareyDownloadInProgress {
            let downloadedCount = rows.filter(\.downloaded).count
            let optionalReady = optionalRows.allSatisfy(\.downloaded)
            if downloadedCount == rows.count {
                modelDownloadStatusMessage = optionalReady
                    ? "all required carey files are downloaded. ScragVAE is ready too."
                    : "all required carey files are downloaded. ScragVAE is optional."
            } else {
                modelDownloadStatusMessage = "\(downloadedCount)/\(rows.count) required carey files are downloaded."
            }
        }
    }

    func startCareyFocusedDownload(targets: [CareyDownloadTarget] = [.base, .sft, .turbo]) {
        guard modelDownloadServiceID == "carey" else { return }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }
        guard let runtime = manager?.services.first(where: { $0.id == "carey" }) else {
            modelDownloadStatusMessage = "carey service is not available in the loaded manifest."
            return
        }
        guard let scriptURL = resolveCareyDownloadScriptURL(for: runtime) else {
            modelDownloadStatusMessage = "focused download script not found."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        activeModelDownloadSessionID = nil
        let normalizedTargets = Array(Set(targets)).sorted { lhs, rhs in
            CareyDownloadTarget.allCases.firstIndex(of: lhs)! < CareyDownloadTarget.allCases.firstIndex(of: rhs)!
        }
        let targetDescription: String
        if Set(normalizedTargets) == Set([.base, .sft, .turbo]) {
            targetDescription = "carey required files"
        } else {
            targetDescription = normalizedTargets.map(\.displayName).joined(separator: " + ")
        }
        activeModelDownloadPath = targetDescription
        isModelDownloadInProgress = true
        isCareyDownloadInProgress = true
        careyActiveDownloadTargets = normalizedTargets
        careyPredownloadProgress = 0
        careyPredownloadActiveLabel = ""
        let requiredModelFiles = requiredCareyModelFiles(for: normalizedTargets)
        careyProgressByLabel = Dictionary(uniqueKeysWithValues: requiredModelFiles.map { ($0.label, 0) })
        modelDownloadStatusMessage = "starting carey download for \(targetDescription)..."

        careyDownloadTask?.cancel()
        careyDownloadTask = Task { [weak self] in
            guard let self else { return }
            let checkpointDirectory = self.resolveCareyCheckpointDirectory(for: runtime)
            var scriptEnvironment = runtime.service.environment
            scriptEnvironment["ACESTEP_CHECKPOINT_DIR"] = checkpointDirectory.path
            let configs = self.activeCareyConfigNames()
            scriptEnvironment["ACESTEP_NO_INIT"] = "true"
            scriptEnvironment["ACESTEP_CONFIG_PATH"] = configs.base
            scriptEnvironment["ACESTEP_BASE_CONFIG_PATH"] = configs.base
            scriptEnvironment["ACESTEP_SFT_CONFIG_PATH"] = configs.sft
            scriptEnvironment["ACESTEP_TURBO_CONFIG_PATH"] = configs.turbo
            scriptEnvironment["ACESTEP_LEGO_CONFIG_PATH"] = configs.lego
            scriptEnvironment["ACESTEP_REGULAR_CONFIG_PATH"] = configs.base
            scriptEnvironment["CAREY_DOWNLOAD_TARGETS"] = normalizedTargets.map(\.rawValue).joined(separator: ",")
            let result = await Task.detached(priority: .userInitiated) {
                Self.runCareyDownloadScript(
                    scriptURL: scriptURL,
                    currentDirectory: runtime.service.workingDirectory,
                    extraEnvironment: scriptEnvironment,
                    onOutputLine: { line in
                        Task { @MainActor [weak self] in
                            self?.handleCareyDownloadOutputLine(line)
                        }
                    }
                )
            }.value

            guard !Task.isCancelled else { return }
            self.isModelDownloadInProgress = false
            self.isCareyDownloadInProgress = false
            if result.exitCode == 0 {
                self.careyPredownloadProgress = 1
                let completedRequiredModelFiles = self.requiredCareyModelFiles(for: normalizedTargets)
                self.careyProgressByLabel = Dictionary(
                    uniqueKeysWithValues: completedRequiredModelFiles.map { ($0.label, 100) }
                )
            }
            self.activeModelDownloadPath = nil
            self.activeModelDownloadSessionID = nil
            self.careyActiveDownloadTargets = []
            self.careyDownloadTask = nil
            self.refreshCareyPredownloadInventory()
            self.modelDownloadStatusMessage = result.message
        }
    }

    private func handleCareyDownloadOutputLine(_ line: String) {
        guard isCareyDownloadInProgress else { return }
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        guard let (label, remainder) = Self.parseCareyProgressLabelAndRemainder(from: trimmed) else {
            if trimmed.hasPrefix("All required") {
                modelDownloadStatusMessage = trimmed
            }
            return
        }

        let normalizedRemainder = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowerRemainder = normalizedRemainder.lowercased()
        careyPredownloadActiveLabel = label
        let requiredModelFiles = currentCareyDownloadFiles()

        if lowerRemainder.hasPrefix("ensuring ") {
            modelDownloadStatusMessage = careyProgressMessage(
                label: label,
                detail: "starting...",
                requiredFiles: requiredModelFiles
            )
            return
        }

        if let percent = Self.parseCareyProgressPercent(from: normalizedRemainder) {
            updateCareyProgress(label: label, percent: percent)
            return
        }

        if lowerRemainder.hasPrefix("complete:")
            || lowerRemainder.hasPrefix("refreshed:")
            || lowerRemainder.hasPrefix("already complete:")
        {
            updateCareyProgress(label: label, percent: 100)
            refreshCareyPredownloadInventory()
            return
        }
    }

    private func updateCareyProgress(label: String, percent: Int) {
        let clampedPercent = max(0, min(100, percent))
        let previous = careyProgressByLabel[label] ?? 0
        careyProgressByLabel[label] = max(previous, clampedPercent)

        let requiredFiles = currentCareyDownloadFiles()
        let totalCount = requiredFiles.count
        let totalPercent = requiredFiles.reduce(0) { partial, item in
            partial + (careyProgressByLabel[item.label] ?? 0)
        }
        careyPredownloadProgress = Double(totalPercent) / Double(totalCount * 100)

        modelDownloadStatusMessage = careyProgressMessage(
            label: label,
            detail: "\(clampedPercent)%",
            requiredFiles: requiredFiles
        )
    }

    private static func parseCareyProgressLabelAndRemainder(from line: String) -> (label: String, remainder: String)? {
        guard line.hasPrefix("["),
              let closingBracket = line.firstIndex(of: "]")
        else {
            return nil
        }

        let labelStart = line.index(after: line.startIndex)
        let label = line[labelStart..<closingBracket]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !label.isEmpty else { return nil }

        let remainderStart = line.index(after: closingBracket)
        let remainder = line[remainderStart...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (label: String(label), remainder: String(remainder))
    }

    private static func parseCareyProgressPercent(from remainder: String) -> Int? {
        let range = NSRange(remainder.startIndex..<remainder.endIndex, in: remainder)
        guard let match = careyProgressPercentRegex.firstMatch(in: remainder, options: [], range: range),
              match.numberOfRanges > 1,
              let percentRange = Range(match.range(at: 1), in: remainder),
              let percent = Int(remainder[percentRange])
        else {
            return nil
        }
        return percent
    }

    private func careyProgressMessage(
        label: String,
        detail: String,
        requiredFiles: [(label: String, relativePath: String)]
    ) -> String {
        let totalCount = requiredFiles.count
        let index = requiredFiles.firstIndex(where: {
            $0.label.caseInsensitiveCompare(label) == .orderedSame
        }).map { $0 + 1 } ?? 0
        if index > 0 {
            return "downloading \(label) (\(index)/\(totalCount)): \(detail)"
        }
        return "downloading \(label): \(detail)"
    }

    func loadCareyModel() {
        runCareyLifecycleAction(
            endpoint: "v1/load",
            successFallbackMessage: "carey model loaded."
        )
    }

    func unloadCareyModel() {
        runCareyLifecycleAction(
            endpoint: "v1/unload",
            successFallbackMessage: "carey model unloaded."
        )
    }

    private func runCareyLifecycleAction(endpoint: String, successFallbackMessage: String) {
        guard modelDownloadServiceID == "carey" else { return }
        guard !isCareyLifecycleActionInProgress else {
            modelDownloadStatusMessage = "carey lifecycle action already in progress."
            return
        }
        guard let baseURL = modelDownloadAPIBaseURL(for: "carey") else {
            modelDownloadStatusMessage = "start carey to run model lifecycle actions."
            return
        }

        isCareyLifecycleActionInProgress = true
        modelDownloadStatusMessage = "sending /\(endpoint)..."

        Task { [weak self] in
            guard let self else { return }
            defer { self.isCareyLifecycleActionInProgress = false }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent(endpoint))
                request.httpMethod = "POST"
                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)

                if
                    let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let responseError = payload["error"] as? String,
                    !responseError.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 10,
                        userInfo: [NSLocalizedDescriptionKey: responseError]
                    )
                }

                if
                    let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let responseData = payload["data"] as? [String: Any],
                    let status = responseData["status"] as? String,
                    !status.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    self.modelDownloadStatusMessage = "carey model status: \(status)."
                } else {
                    self.modelDownloadStatusMessage = successFallbackMessage
                }
            } catch {
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    func startModelDownload(_ modelPath: String) {
        let serviceID = modelDownloadServiceID
        let serviceDisplayName = modelDownloadDisplayName(forServiceID: serviceID)
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            if serviceID == "audiocraft_mlx", canRunGaryOfflineDownloads {
                startGaryOfflineModelDownload(modelPath)
                return
            }
            if serviceID == "foundation", canRunFoundationOfflineDownloads {
                startFoundationOfflineModelDownload(modelPath)
                return
            }
            if serviceID == "melodyflow", canRunMelodyflowOfflineDownloads {
                startMelodyflowOfflineModelDownload(modelPath)
                return
            }
            modelDownloadStatusMessage = {
                switch serviceID {
                case "audiocraft_mlx":
                    return "build gary before downloading models."
                case "foundation":
                    return "build foundation-1 before downloading models."
                case "melodyflow":
                    return "build terry (melodyflow) before downloading models."
                default:
                    return "start \(serviceDisplayName) to download models."
                }
            }()
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        setModelDownloadState(
            for: modelPath,
            isDownloading: true,
            downloaded: false,
            progress: 0,
            statusMessage: "starting download..."
        )
        isModelDownloadInProgress = true
        activeModelDownloadPath = modelPath
        modelDownloadStatusMessage = "starting \(modelPath)..."

        Task { [weak self] in
            guard let self else { return }
            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("api/models/predownload"))
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: ["model_name": modelPath])

                let (data, response) = try await URLSession.shared.data(for: request)
                try self.ensureHTTP200(response: response, body: data)
                let decoder = JSONDecoder()
                let startResponse = try decoder.decode(RemotePredownloadStartResponse.self, from: data)
                guard startResponse.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 3,
                        userInfo: [NSLocalizedDescriptionKey: "unable to start model download."]
                    )
                }

                self.activeModelDownloadSessionID = startResponse.sessionID
                self.modelDownloadStatusMessage = startResponse.message ?? "downloading \(modelPath)..."
                self.startModelDownloadPolling(
                    sessionID: startResponse.sessionID,
                    modelPath: modelPath,
                    serviceID: serviceID,
                    baseURL: baseURL
                )
            } catch {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                self.setModelDownloadState(
                    for: modelPath,
                    isDownloading: false,
                    downloaded: false,
                    progress: 0,
                    statusMessage: "download failed"
                )
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    func saveStableAudioToken() {
        let token = stableAudioTokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            stableAudioTokenStatus = "paste your hugging face token first."
            return
        }

        stableAudioTokenStatus = "saving token..."
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                try StableAudioAuthKeychain.saveToken(token)
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.sharedHuggingFaceToken = token
                    self.stableAudioTokenInput = ""
                    self.applyStableAudioTokenState(configured: true)
                    let restartedServices = self.manager?.restartServicesUsingSharedHuggingFaceToken() ?? []
                    if restartedServices.isEmpty {
                        self.stableAudioTokenStatus = "token saved in keychain."
                    } else {
                        let label = restartedServices.joined(separator: ", ")
                        self.stableAudioTokenStatus = "token saved in keychain. restarting \(label)..."
                    }
                }
            } catch {
                let message = error.localizedDescription
                DispatchQueue.main.async { [weak self] in
                    self?.stableAudioTokenStatus = message
                }
            }
        }
    }

    func clearStableAudioToken() {
        stableAudioTokenStatus = "removing token..."
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                try StableAudioAuthKeychain.deleteToken()
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.sharedHuggingFaceToken = nil
                    self.stableAudioTokenInput = ""
                    self.applyStableAudioTokenState(configured: false)
                    self.stableAudioTokenStatus = "saved token removed."
                }
            } catch {
                let message = error.localizedDescription
                DispatchQueue.main.async { [weak self] in
                    self?.stableAudioTokenStatus = message
                }
            }
        }
    }

    func refreshStableAudioTokenState() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let token = StableAudioAuthKeychain.readToken()
            let configured = token?.isEmpty == false
            DispatchQueue.main.async { [weak self] in
                self?.sharedHuggingFaceToken = token
                self?.applyStableAudioTokenState(configured: configured)
            }
        }
    }

    private func reloadHFScreenshots() {
        guard !manifestPath.isEmpty else {
            stableAudioStep2ScreenshotPath = nil
            return
        }

        guard let screenshotDirectory = findScreenshotDirectory(startingAt: manifestPath) else {
            stableAudioStep2ScreenshotPath = nil
            return
        }

        let fileManager = FileManager.default
        let allowedExtensions = Set(["png", "jpg", "jpeg", "webp"])

        let paths = (try? fileManager.contentsOfDirectory(
            at: screenshotDirectory,
            includingPropertiesForKeys: nil
        ))?.filter { url in
            allowedExtensions.contains(url.pathExtension.lowercased())
        }
        .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        .map(\.path) ?? []

        stableAudioStep2ScreenshotPath = preferredStep2Screenshot(from: paths)
    }

    private func bindManager(_ manager: ServiceManager) {
        managerCancellables.removeAll()

        manager.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &managerCancellables)

        manager.$latestRebuildFailure
            .receive(on: DispatchQueue.main)
            .sink { [weak self] report in
                guard let self else { return }
                self.rebuildFailureReport = report
                if report == nil {
                    self.rebuildDiagnosticsStatusMessage = ""
                }
            }
            .store(in: &managerCancellables)
    }

    func diagnosticsReportText(for report: RebuildFailureReport) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]

        var lines: [String] = []
        lines.append("gary4local rebuild diagnostics")
        lines.append("timestamp: \(formatter.string(from: report.createdAt))")
        lines.append("service_id: \(report.serviceID)")
        lines.append("service_name: \(report.serviceName)")
        lines.append("summary: \(report.summary)")
        lines.append("manifest_path: \(manifestPath)")
        lines.append("working_directory: \(report.workingDirectory.path)")
        lines.append("log_file: \(report.logFile.path)")
        if let pythonExecutable = report.pythonExecutable {
            lines.append("python_executable: \(pythonExecutable)")
        }
        if let requirementsFile = report.requirementsFile {
            lines.append("requirements_file: \(requirementsFile.path)")
            if let editableRequirementsFile = resolvedRequirementsFile(for: report),
               editableRequirementsFile.path != requirementsFile.path {
                lines.append("editable_requirements_file: \(editableRequirementsFile.path)")
            }
        }
        if let venvDirectory = report.venvDirectory {
            lines.append("venv_directory: \(venvDirectory.path)")
        }
        lines.append("")
        lines.append("---- recent log tail ----")
        if report.logTail.isEmpty {
            lines.append("(no log output)")
        } else {
            lines.append(report.logTail)
        }

        return lines.joined(separator: "\n")
    }

    private func resolvedRequirementsFile(for report: RebuildFailureReport) -> URL? {
        guard let requirementsFile = report.requirementsFile else {
            return nil
        }
        return resolveLeafRequirementsFile(startingAt: requirementsFile)
    }

    private func resolveLeafRequirementsFile(startingAt root: URL, maxDepth: Int = 6) -> URL {
        var current = root.standardizedFileURL
        let fileManager = FileManager.default

        for _ in 0..<maxDepth {
            guard fileManager.fileExists(atPath: current.path),
                  let contents = try? String(contentsOf: current, encoding: .utf8) else {
                return current
            }

            let parsed = parseRequirements(contents)
            guard !parsed.includePaths.isEmpty, !parsed.hasDirectPackages else {
                return current
            }

            guard let firstIncludePath = parsed.includePaths.first else {
                return current
            }

            if firstIncludePath.contains("://") {
                return current
            }

            let next: URL
            if firstIncludePath.hasPrefix("/") {
                next = URL(fileURLWithPath: firstIncludePath).standardizedFileURL
            } else {
                next = current
                    .deletingLastPathComponent()
                    .appendingPathComponent(firstIncludePath)
                    .standardizedFileURL
            }

            guard fileManager.fileExists(atPath: next.path), next.path != current.path else {
                return current
            }
            current = next
        }

        return current
    }

    private func parseRequirements(_ contents: String) -> (includePaths: [String], hasDirectPackages: Bool) {
        var includePaths: [String] = []
        var hasDirectPackages = false

        for rawLine in contents.split(whereSeparator: \.isNewline) {
            var line = String(rawLine).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }

            if let hashIndex = line.firstIndex(of: "#") {
                line = String(line[..<hashIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
                if line.isEmpty { continue }
            }

            let tokens = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
            guard let first = tokens.first else { continue }

            if first == "-r" || first == "--requirement" {
                if tokens.count >= 2 {
                    includePaths.append(cleanRequirementsPathToken(tokens[1]))
                }
                continue
            }
            if first.hasPrefix("-r"), first != "-r" {
                includePaths.append(cleanRequirementsPathToken(String(first.dropFirst(2))))
                continue
            }
            if first.hasPrefix("--requirement=") {
                includePaths.append(
                    cleanRequirementsPathToken(
                        String(first.dropFirst("--requirement=".count))
                    )
                )
                continue
            }

            if first == "-c" || first == "--constraint" ||
                first.hasPrefix("-c") || first.hasPrefix("--constraint=") {
                continue
            }

            hasDirectPackages = true
            break
        }

        return (includePaths: includePaths, hasDirectPackages: hasDirectPackages)
    }

    private func cleanRequirementsPathToken(_ token: String) -> String {
        var cleaned = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.hasPrefix("\""), cleaned.hasSuffix("\""), cleaned.count >= 2 {
            cleaned.removeFirst()
            cleaned.removeLast()
        } else if cleaned.hasPrefix("'"), cleaned.hasSuffix("'"), cleaned.count >= 2 {
            cleaned.removeFirst()
            cleaned.removeLast()
        }
        return cleaned
    }

    private func preferredStep2Screenshot(from paths: [String]) -> String? {
        if let preferred = paths.first(where: { $0.contains("6.51.30") }) {
            return preferred
        }
        return paths.first
    }

    private func findScreenshotDirectory(startingAt manifestPath: String) -> URL? {
        var cursor = URL(fileURLWithPath: manifestPath).deletingLastPathComponent()
        let fileManager = FileManager.default

        while cursor.path != "/" {
            let candidate = cursor.appendingPathComponent("hf-screenshots")
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
               isDirectory.boolValue {
                return candidate
            }
            cursor.deleteLastPathComponent()
        }
        return nil
    }

    private func startLogRefreshLoop() {
        logRefreshTask?.cancel()
        let interval = logRefreshIntervalNanoseconds
        logRefreshTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: interval)
                await MainActor.run {
                    self?.requestLogRefresh(force: false)
                }
            }
        }
    }

    private func requestLogRefresh(force: Bool) {
        guard force || isLogViewerPinnedToBottom else {
            return
        }

        guard !isLogRefreshInFlight else {
            pendingForcedLogRefresh = pendingForcedLogRefresh || force
            return
        }

        guard let manager, let selectedServiceID,
              let runtime = manager.services.first(where: { $0.id == selectedServiceID }) else {
            selectedLogText = ""
            return
        }

        let logFile = runtime.service.logFile
        let previousMetadata = force ? nil : lastLogMetadataByService[selectedServiceID]

        isLogRefreshInFlight = true
        Task { [weak self] in
            let snapshot = await Task.detached(priority: .utility) {
                ServiceManager.readLogTailSnapshot(
                    at: logFile,
                    maxLines: 220,
                    maxBytes: 192_000
                )
            }.value

            await MainActor.run {
                guard let self else { return }
                defer {
                    self.isLogRefreshInFlight = false
                    let shouldForceRefresh = self.pendingForcedLogRefresh
                    self.pendingForcedLogRefresh = false
                    if shouldForceRefresh {
                        self.requestLogRefresh(force: true)
                    }
                }

                guard self.selectedServiceID == selectedServiceID else {
                    return
                }

                let metadata = LogMetadata(
                    fileSize: snapshot.fileSize,
                    modificationDate: snapshot.modificationDate
                )

                if !force, let previousMetadata, previousMetadata == metadata {
                    return
                }

                self.lastLogMetadataByService[selectedServiceID] = metadata
                if self.selectedLogText != snapshot.text {
                    self.selectedLogText = snapshot.text
                }
            }
        }
    }

    private func startModelDownloadPolling(
        sessionID: String,
        modelPath: String,
        serviceID: String,
        baseURL: URL,
        statusPathPrefix: String = "api/models/predownload_status"
    ) {
        modelDownloadPollTask?.cancel()
        let statusURL = baseURL
            .appendingPathComponent(statusPathPrefix)
            .appendingPathComponent(sessionID)
        modelDownloadPollTask = Task { [weak self] in
            guard let self else { return }
            let decoder = JSONDecoder()

            while !Task.isCancelled {
                do {
                    let (data, response) = try await URLSession.shared.data(from: statusURL)
                    try self.ensureHTTP200(response: response, body: data)
                    let pollResponse = try decoder.decode(RemotePredownloadStatusResponse.self, from: data)
                    if !pollResponse.success {
                        throw NSError(
                            domain: "ControlCenterViewModel",
                            code: 4,
                            userInfo: [NSLocalizedDescriptionKey: "model download polling failed."]
                        )
                    }

                    let normalizedProgress = self.derivedPredownloadProgress(from: pollResponse)
                    let progress = Double(normalizedProgress) / 100.0

                    let queueMessage = pollResponse.queueStatus?.message?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    let fallbackMessage: String
                    switch pollResponse.status {
                    case "completed":
                        fallbackMessage = "downloaded"
                    case "failed":
                        fallbackMessage = pollResponse.error ?? "download failed"
                    case "warming", "processing":
                        fallbackMessage = "downloading..."
                    default:
                        fallbackMessage = pollResponse.status
                    }
                    let statusMessage = queueMessage.isEmpty ? fallbackMessage : queueMessage

                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: pollResponse.status == "warming" || pollResponse.status == "processing",
                        downloaded: pollResponse.status == "completed",
                        progress: progress,
                        statusMessage: statusMessage
                    )
                    if serviceID == "stable_audio" {
                        self.stableAudioPredownloadProgress = progress
                    } else if serviceID == "sa3" {
                        self.sa3PredownloadProgress = progress
                    }
                    self.modelDownloadStatusMessage = statusMessage

                    if pollResponse.status == "completed" {
                        self.isModelDownloadInProgress = false
                        self.activeModelDownloadPath = nil
                        self.activeModelDownloadSessionID = nil
                        self.modelDownloadPollTask = nil
                        if self.modelDownloadServiceID == serviceID,
                           serviceID != "stable_audio",
                           serviceID != "sa3" {
                            self.refreshModelCatalogAndStatuses()
                        } else if serviceID == "sa3" {
                            self.refreshSA3PredownloadInventory()
                        } else if serviceID == "stable_audio" {
                            self.refreshStableAudioPredownloadInventory(
                                checkpointsHint: self.stableAudioPredownloadCheckpoints
                            )
                        }
                        return
                    }

                    if pollResponse.status == "failed" {
                        self.isModelDownloadInProgress = false
                        self.activeModelDownloadPath = nil
                        self.activeModelDownloadSessionID = nil
                        self.modelDownloadPollTask = nil
                        return
                    }
                } catch {
                    self.isModelDownloadInProgress = false
                    self.activeModelDownloadPath = nil
                    self.activeModelDownloadSessionID = nil
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: false,
                        progress: 0,
                        statusMessage: "download polling failed"
                    )
                    if serviceID == "stable_audio" {
                        self.stableAudioPredownloadProgress = 0
                    } else if serviceID == "sa3" {
                        self.sa3PredownloadProgress = 0
                    }
                    self.modelDownloadStatusMessage = error.localizedDescription
                    self.modelDownloadPollTask = nil
                    return
                }

                try? await Task.sleep(nanoseconds: self.modelDownloadPollIntervalNanoseconds)
            }
        }
    }

    private func derivedPredownloadProgress(from response: RemotePredownloadStatusResponse) -> Int {
        if response.status == "completed" {
            return 100
        }

        var normalized = max(0, min(100, response.progress))
        guard
            let queue = response.queueStatus,
            let stageIndex = queue.stageIndex,
            let stageTotal = queue.stageTotal,
            stageIndex > 0,
            stageTotal > 0
        else {
            return normalized
        }

        let stagePercent = max(0, min(100, queue.downloadPercent ?? 0))
        let derivedRaw: Double
        if modelDownloadServiceID == "stable_audio", stageTotal == 5 {
            // Match backend weighting so checkpoint transfer drives visible progress.
            let primaryStageWeight = 0.96
            if stageIndex <= 1 {
                derivedRaw = (Double(stagePercent) / 100.0) * primaryStageWeight * 100.0
            } else {
                let secondaryStageWeight = (1.0 - primaryStageWeight) / 4.0
                let completedSecondaryStages = max(0, stageIndex - 2)
                derivedRaw = (
                    primaryStageWeight
                    + (Double(completedSecondaryStages) * secondaryStageWeight)
                    + ((Double(stagePercent) / 100.0) * secondaryStageWeight)
                ) * 100.0
            }
        } else if modelDownloadServiceID == "foundation", stageTotal == 2 {
            // Foundation's safetensors payload is effectively the whole download.
            let primaryStageWeight = 0.99
            if stageIndex <= 1 {
                derivedRaw = (Double(stagePercent) / 100.0) * primaryStageWeight * 100.0
            } else {
                let secondaryStageWeight = 1.0 - primaryStageWeight
                derivedRaw = (
                    primaryStageWeight
                    + ((Double(stagePercent) / 100.0) * secondaryStageWeight)
                ) * 100.0
            }
        } else {
            derivedRaw = (
                (Double(stageIndex - 1) + (Double(stagePercent) / 100.0))
                / Double(stageTotal)
            ) * 100.0
        }

        var derived = Int(derivedRaw.rounded(.up))
        derived = max(0, min(99, derived))
        if stagePercent > 0 {
            derived = max(1, derived)
        }
        normalized = max(normalized, derived)
        return normalized
    }

    private func flattenRemoteModels(_ source: [String: [RemoteModelEntry]]) -> [DownloadableModel] {
        let sizeOrder = ["small", "medium", "large"]
        var flattened: [DownloadableModel] = []

        for size in sizeOrder {
            guard let entries = source[size] else { continue }
            for entry in entries {
                if entry.type == "single", let path = entry.path {
                    flattened.append(
                        DownloadableModel(
                            id: path,
                            size: size,
                            displayName: entry.name,
                            path: path,
                            downloaded: false,
                            isDownloading: false,
                            progress: 0,
                            statusMessage: "not downloaded"
                        )
                    )
                    continue
                }

                guard entry.type == "group", let checkpoints = entry.checkpoints else {
                    continue
                }
                for checkpoint in checkpoints {
                    flattened.append(
                        DownloadableModel(
                            id: checkpoint.path,
                            size: size,
                            displayName: checkpoint.name,
                            path: checkpoint.path,
                            downloaded: false,
                            isDownloading: false,
                            progress: 0,
                            statusMessage: "not downloaded"
                        )
                    )
                }
            }
        }

        return flattened
    }

    private func setModelDownloadState(
        for modelPath: String,
        isDownloading: Bool,
        downloaded: Bool,
        progress: Double,
        statusMessage: String
    ) {
        guard let index = downloadableModels.firstIndex(where: { $0.path == modelPath }) else { return }
        downloadableModels[index].isDownloading = isDownloading
        downloadableModels[index].downloaded = downloaded
        downloadableModels[index].progress = max(0, min(1, progress))
        downloadableModels[index].statusMessage = statusMessage
    }

    private func refreshFoundationOfflineModelCatalogAndStatuses() {
        guard let runtime = currentFoundationRuntime(),
              let helperURL = foundationPredownloadHelperURL(for: runtime)
        else {
            downloadableModels = []
            modelDownloadStatusMessage = "build foundation-1 before managing model downloads."
            isModelCatalogLoading = false
            return
        }

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = runtime.service.environment

        isModelCatalogLoading = true
        modelDownloadStatusMessage = "loading model catalog..."

        Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.loadLocalPredownloadCatalogOutputs(
                    executableURL: executableURL,
                    helperURL: helperURL,
                    currentDirectory: workingDirectory,
                    environment: environment
                )
            }.value

            guard !Task.isCancelled else { return }
            self.applyOfflinePredownloadCatalogResult(
                result,
                expectedHelperName: "foundation helper"
            )
        }
    }

    private func refreshGaryOfflineModelCatalogAndStatuses() {
        guard let runtime = currentGaryRuntime(),
              let helperURL = garyPredownloadHelperURL(for: runtime)
        else {
            downloadableModels = []
            modelDownloadStatusMessage = "build gary before managing model downloads."
            isModelCatalogLoading = false
            return
        }

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = garyOfflinePredownloadEnvironment(for: runtime)

        isModelCatalogLoading = true
        modelDownloadStatusMessage = "loading model catalog..."

        Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.loadLocalPredownloadCatalogOutputs(
                    executableURL: executableURL,
                    helperURL: helperURL,
                    currentDirectory: workingDirectory,
                    environment: environment
                )
            }.value

            guard !Task.isCancelled else { return }
            self.applyOfflinePredownloadCatalogResult(
                result,
                expectedHelperName: "gary helper"
            )
        }
    }

    private func refreshMelodyflowOfflineModelCatalogAndStatuses() {
        guard let runtime = currentMelodyflowRuntime(),
              let helperURL = melodyflowPredownloadHelperURL(for: runtime)
        else {
            downloadableModels = []
            modelDownloadStatusMessage = "build terry (melodyflow) before managing model downloads."
            isModelCatalogLoading = false
            return
        }

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = runtime.service.environment

        isModelCatalogLoading = true
        modelDownloadStatusMessage = "loading model catalog..."

        Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.loadLocalPredownloadCatalogOutputs(
                    executableURL: executableURL,
                    helperURL: helperURL,
                    currentDirectory: workingDirectory,
                    environment: environment
                )
            }.value

            guard !Task.isCancelled else { return }
            self.applyOfflinePredownloadCatalogResult(
                result,
                expectedHelperName: "melodyflow helper"
            )
        }
    }

    private func fetchStableAudioOfflinePredownloadCheckpoints() {
        guard let runtime = currentStableAudioRuntime(),
              let helperURL = stableAudioPredownloadHelperURL(for: runtime)
        else {
            modelDownloadStatusMessage = "build jerry before fetching checkpoints."
            return
        }

        let repo = stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty else {
            modelDownloadStatusMessage = "enter a hugging face repo first."
            return
        }
        guard !isStableAudioCheckpointFetchInProgress else { return }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = stableAudioOfflinePredownloadEnvironment(for: runtime)

        isStableAudioCheckpointFetchInProgress = true
        modelDownloadStatusMessage = "fetching checkpoints from \(repo)..."

        Task { [weak self] in
            guard let self else { return }
            defer { self.isStableAudioCheckpointFetchInProgress = false }

            let result = await Task.detached(priority: .userInitiated) {
                Self.runLocalProcess(
                    executableURL: executableURL,
                    arguments: [
                        helperURL.path,
                        "checkpoints",
                        "--finetune-repo",
                        repo,
                    ],
                    currentDirectory: workingDirectory,
                    environment: environment
                )
            }.value

            guard !Task.isCancelled else { return }

            do {
                guard result.exitCode == 0 else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 7001,
                        userInfo: [NSLocalizedDescriptionKey: result.output]
                    )
                }

                let payload = try JSONDecoder().decode(
                    StableAudioCheckpointsResponse.self,
                    from: Self.jsonDataFromProcessOutput(result.output)
                )
                guard payload.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 7002,
                        userInfo: [
                            NSLocalizedDescriptionKey: payload.error ?? "failed to fetch checkpoints."
                        ]
                    )
                }

                let checkpoints = (payload.checkpoints ?? []).sorted {
                    $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
                }
                self.stableAudioPredownloadCheckpoints = checkpoints
                self.stableAudioPredownloadSelectedCheckpoint = checkpoints.first ?? ""
                self.stableAudioPredownloadCheckpointDownloaded = [:]
                if checkpoints.isEmpty {
                    self.modelDownloadStatusMessage = "no .ckpt files found in \(repo)."
                } else {
                    self.modelDownloadStatusMessage = "\(checkpoints.count) checkpoint\(checkpoints.count == 1 ? "" : "s") found."
                }
                self.refreshStableAudioPredownloadInventory(checkpointsHint: checkpoints)
            } catch {
                self.stableAudioPredownloadCheckpoints = []
                self.stableAudioPredownloadCheckpointDownloaded = [:]
                self.stableAudioPredownloadSelectedCheckpoint = ""
                self.modelDownloadStatusMessage = error.localizedDescription
            }
        }
    }

    private func refreshStableAudioOfflinePredownloadInventory(checkpointsHint: [String] = []) {
        guard let runtime = currentStableAudioRuntime(),
              let helperURL = stableAudioPredownloadHelperURL(for: runtime)
        else {
            stableAudioInventoryModels = []
            stableAudioCachedFinetunes = []
            stableAudioPredownloadCheckpointDownloaded = [:]
            return
        }

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = stableAudioOfflinePredownloadEnvironment(for: runtime)
        let repo = stableAudioPredownloadRepoInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let hints = checkpointsHint.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                var arguments = [helperURL.path, "inventory"]
                if !repo.isEmpty {
                    arguments.append(contentsOf: ["--finetune-repo", repo])
                }
                for checkpoint in hints {
                    arguments.append(contentsOf: ["--checkpoint", checkpoint])
                }
                return Self.runLocalProcess(
                    executableURL: executableURL,
                    arguments: arguments,
                    currentDirectory: workingDirectory,
                    environment: environment
                )
            }.value

            guard !Task.isCancelled else { return }

            do {
                guard result.exitCode == 0 else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 7003,
                        userInfo: [NSLocalizedDescriptionKey: result.output]
                    )
                }

                let payload = try JSONDecoder().decode(
                    StableAudioPredownloadInventoryResponse.self,
                    from: Self.jsonDataFromProcessOutput(result.output)
                )
                guard payload.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 7004,
                        userInfo: [
                            NSLocalizedDescriptionKey: payload.error ?? "failed to load stable inventory."
                        ]
                    )
                }

                self.stableAudioInventoryModels = payload.knownModels.map { row in
                    StableAudioInventoryModelStatus(
                        id: row.repoID,
                        label: row.label,
                        downloaded: row.downloaded,
                        missing: row.missing
                    )
                }
                self.stableAudioCachedFinetunes = payload.cachedFinetunes
                self.stableAudioPredownloadCheckpointDownloaded = Dictionary(
                    uniqueKeysWithValues: payload.finetuneCheckpoints.map { ($0.name, $0.downloaded) }
                )
            } catch {
                if self.stableAudioInventoryModels.isEmpty {
                    self.modelDownloadStatusMessage = error.localizedDescription
                }
            }
        }
    }

    private func startFoundationOfflineModelDownload(_ modelPath: String) {
        guard let runtime = currentFoundationRuntime(),
              let helperURL = foundationPredownloadHelperURL(for: runtime)
        else {
            modelDownloadStatusMessage = "build foundation-1 before downloading models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        foundationLocalDownloadTask?.cancel()
        foundationLocalDownloadTask = nil

        setModelDownloadState(
            for: modelPath,
            isDownloading: true,
            downloaded: false,
            progress: 0,
            statusMessage: "starting download..."
        )
        isModelDownloadInProgress = true
        activeModelDownloadPath = modelPath
        activeModelDownloadSessionID = "local-foundation-predownload"
        modelDownloadStatusMessage = "starting \(modelPath)..."

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = runtime.service.environment

        foundationLocalDownloadTask = Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.runStreamingLocalProcess(
                    executableURL: executableURL,
                    arguments: [
                        helperURL.path,
                        "download",
                        "--model-name",
                        modelPath,
                    ],
                    currentDirectory: workingDirectory,
                    environment: environment,
                    onOutputLine: { line in
                        Task { @MainActor [weak self] in
                            self?.handleFoundationOfflineDownloadOutputLine(line, modelPath: modelPath)
                        }
                    }
                )
            }.value

            guard !Task.isCancelled else { return }

            if self.isModelDownloadInProgress {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                if result.exitCode == 0 {
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: true,
                        progress: 1,
                        statusMessage: "downloaded"
                    )
                    self.modelDownloadStatusMessage = "foundation-1 model files downloaded."
                    self.refreshFoundationOfflineModelCatalogAndStatuses()
                } else {
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: false,
                        progress: 0,
                        statusMessage: "download failed"
                    )
                    self.modelDownloadStatusMessage = result.output.isEmpty
                        ? "foundation-1 download failed."
                        : result.output
                }
            }

            self.foundationLocalDownloadTask = nil
        }
    }

    private func startGaryOfflineModelDownload(_ modelPath: String) {
        guard let runtime = currentGaryRuntime(),
              let helperURL = garyPredownloadHelperURL(for: runtime)
        else {
            modelDownloadStatusMessage = "build gary before downloading models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        garyLocalDownloadTask?.cancel()
        garyLocalDownloadTask = nil

        setModelDownloadState(
            for: modelPath,
            isDownloading: true,
            downloaded: false,
            progress: 0,
            statusMessage: "starting download..."
        )
        isModelDownloadInProgress = true
        activeModelDownloadPath = modelPath
        activeModelDownloadSessionID = "local-gary-predownload"
        modelDownloadStatusMessage = "starting \(modelPath)..."

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = garyOfflinePredownloadEnvironment(for: runtime)

        garyLocalDownloadTask = Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.runStreamingLocalProcess(
                    executableURL: executableURL,
                    arguments: [
                        helperURL.path,
                        "download",
                        "--model-name",
                        modelPath,
                    ],
                    currentDirectory: workingDirectory,
                    environment: environment,
                    onOutputLine: { line in
                        Task { @MainActor [weak self] in
                            self?.handleOfflineHelperDownloadOutputLine(
                                line,
                                modelPath: modelPath
                            )
                        }
                    }
                )
            }.value

            guard !Task.isCancelled else { return }

            if self.isModelDownloadInProgress {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                if result.exitCode == 0 {
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: true,
                        progress: 1,
                        statusMessage: "downloaded"
                    )
                    self.modelDownloadStatusMessage = "gary model files downloaded."
                    self.refreshGaryOfflineModelCatalogAndStatuses()
                } else {
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: false,
                        progress: 0,
                        statusMessage: "download failed"
                    )
                    self.modelDownloadStatusMessage = result.output.isEmpty
                        ? "gary download failed."
                        : result.output
                }
            }

            self.garyLocalDownloadTask = nil
        }
    }

    private func startMelodyflowOfflineModelDownload(_ modelPath: String) {
        guard let runtime = currentMelodyflowRuntime(),
              let helperURL = melodyflowPredownloadHelperURL(for: runtime)
        else {
            modelDownloadStatusMessage = "build terry (melodyflow) before downloading models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        melodyflowLocalDownloadTask?.cancel()
        melodyflowLocalDownloadTask = nil

        setModelDownloadState(
            for: modelPath,
            isDownloading: true,
            downloaded: false,
            progress: 0,
            statusMessage: "starting download..."
        )
        isModelDownloadInProgress = true
        activeModelDownloadPath = modelPath
        activeModelDownloadSessionID = "local-melodyflow-predownload"
        modelDownloadStatusMessage = "starting \(modelPath)..."

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = runtime.service.environment

        melodyflowLocalDownloadTask = Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.runStreamingLocalProcess(
                    executableURL: executableURL,
                    arguments: [
                        helperURL.path,
                        "download",
                        "--model-name",
                        modelPath,
                    ],
                    currentDirectory: workingDirectory,
                    environment: environment,
                    onOutputLine: { line in
                        Task { @MainActor [weak self] in
                            self?.handleOfflineHelperDownloadOutputLine(
                                line,
                                modelPath: modelPath
                            )
                        }
                    }
                )
            }.value

            guard !Task.isCancelled else { return }

            if self.isModelDownloadInProgress {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                if result.exitCode == 0 {
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: true,
                        progress: 1,
                        statusMessage: "downloaded"
                    )
                    self.modelDownloadStatusMessage = "melodyflow model files downloaded."
                    self.refreshMelodyflowOfflineModelCatalogAndStatuses()
                } else {
                    self.setModelDownloadState(
                        for: modelPath,
                        isDownloading: false,
                        downloaded: false,
                        progress: 0,
                        statusMessage: "download failed"
                    )
                    self.modelDownloadStatusMessage = result.output.isEmpty
                        ? "melodyflow download failed."
                        : result.output
                }
            }

            self.melodyflowLocalDownloadTask = nil
        }
    }

    private func startStableAudioOfflinePredownload(payload: [String: Any], targetLabel: String) {
        guard let runtime = currentStableAudioRuntime(),
              let helperURL = stableAudioPredownloadHelperURL(for: runtime)
        else {
            modelDownloadStatusMessage = "build jerry before pre-downloading models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        let targetType = (payload["target_type"] as? String ?? "pretrained")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = stableAudioOfflinePredownloadEnvironment(for: runtime)

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        stableAudioLocalDownloadTask?.cancel()
        stableAudioLocalDownloadTask = nil

        activeModelDownloadPath = targetLabel
        activeModelDownloadSessionID = "local-stable-audio-predownload"
        stableAudioPredownloadTargetLabel = targetLabel
        stableAudioPredownloadProgress = 0
        isModelDownloadInProgress = true
        modelDownloadStatusMessage = "starting \(targetLabel)..."

        stableAudioLocalDownloadTask = Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                var arguments: [String]
                if targetType == "finetune" {
                    arguments = [
                        helperURL.path,
                        "download-finetune",
                        "--finetune-repo",
                        String(describing: payload["finetune_repo"] ?? ""),
                        "--finetune-checkpoint",
                        String(describing: payload["finetune_checkpoint"] ?? ""),
                        "--base-repo",
                        String(describing: payload["base_repo"] ?? "stabilityai/stable-audio-open-small"),
                    ]
                } else {
                    arguments = [
                        helperURL.path,
                        "download-pretrained",
                        "--repo-id",
                        String(describing: payload["repo_id"] ?? "stabilityai/stable-audio-open-small"),
                    ]
                }
                if (payload["require_token"] as? Bool) == true {
                    arguments.append("--require-token")
                }

                return Self.runStreamingLocalProcess(
                    executableURL: executableURL,
                    arguments: arguments,
                    currentDirectory: workingDirectory,
                    environment: environment,
                    onOutputLine: { line in
                        Task { @MainActor [weak self] in
                            self?.handleOfflineHelperDownloadOutputLine(line, modelPath: targetLabel)
                        }
                    }
                )
            }.value

            guard !Task.isCancelled else { return }

            if self.isModelDownloadInProgress {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                if result.exitCode == 0 {
                    self.stableAudioPredownloadProgress = 1
                    self.modelDownloadStatusMessage = "\(targetLabel) is ready for offline use."
                    self.refreshStableAudioPredownloadInventory(
                        checkpointsHint: self.stableAudioPredownloadCheckpoints
                    )
                } else {
                    self.stableAudioPredownloadProgress = 0
                    self.modelDownloadStatusMessage = result.output.isEmpty
                        ? "stable audio download failed."
                        : Self.userFacingStableAudioPredownloadFailureMessage(from: result.output)
                }
            }

            self.stableAudioLocalDownloadTask = nil
        }
    }

    private func handleFoundationOfflineDownloadOutputLine(_ line: String, modelPath: String) {
        handleOfflineHelperDownloadOutputLine(line, modelPath: modelPath)
    }

    private func refreshSA3OfflinePredownloadInventory() {
        guard let runtime = currentSA3Runtime(),
              let helperURL = sa3PredownloadHelperURL(for: runtime)
        else {
            sa3InventoryModels = []
            return
        }

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = sa3OfflinePredownloadEnvironment(for: runtime)

        Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.runLocalProcess(
                    executableURL: executableURL,
                    arguments: [helperURL.path, "inventory"],
                    currentDirectory: workingDirectory,
                    environment: environment
                )
            }.value

            guard !Task.isCancelled else { return }

            guard result.exitCode == 0 else {
                if self.sa3InventoryModels.isEmpty {
                    self.modelDownloadStatusMessage = result.output.isEmpty
                        ? "failed to load sa3 inventory."
                        : result.output
                }
                return
            }

            do {
                let payload = try JSONDecoder().decode(
                    SA3PredownloadInventoryResponse.self,
                    from: Self.jsonDataFromProcessOutput(result.output)
                )
                guard payload.success else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 11,
                        userInfo: [NSLocalizedDescriptionKey: payload.error ?? "failed to load sa3 inventory."]
                    )
                }

                self.sa3InventoryModels = payload.knownModels.map { row in
                    SA3InventoryModelStatus(
                        repoID: row.repoID,
                        label: row.label,
                        downloaded: row.downloaded,
                        missing: row.missing
                    )
                }
            } catch {
                if self.sa3InventoryModels.isEmpty {
                    self.modelDownloadStatusMessage = error.localizedDescription
                }
            }
        }
    }

    private func startSA3OfflinePredownloadRequiredModels() {
        guard let runtime = currentSA3Runtime(),
              let helperURL = sa3PredownloadHelperURL(for: runtime)
        else {
            modelDownloadStatusMessage = "build sa3 before pre-downloading models."
            return
        }
        guard !isModelDownloadInProgress else {
            modelDownloadStatusMessage = "a model download is already running."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        sa3LocalDownloadTask?.cancel()
        sa3LocalDownloadTask = nil
        activeModelDownloadSessionID = nil
        activeModelDownloadPath = "required sa3 models"
        sa3PredownloadTargetLabel = "required sa3 models"
        sa3PredownloadProgress = 0
        isModelDownloadInProgress = true
        modelDownloadStatusMessage = "starting required sa3 models..."

        let executableURL = runtime.service.executable
        let workingDirectory = runtime.service.workingDirectory
        let environment = sa3OfflinePredownloadEnvironment(for: runtime)

        sa3LocalDownloadTask = Task { [weak self] in
            guard let self else { return }
            let result = await Task.detached(priority: .userInitiated) {
                Self.runStreamingLocalProcess(
                    executableURL: executableURL,
                    arguments: [helperURL.path, "download-required"],
                    currentDirectory: workingDirectory,
                    environment: environment,
                    onOutputLine: { line in
                        Task { @MainActor [weak self] in
                            self?.handleOfflineHelperDownloadOutputLine(
                                line,
                                modelPath: "required sa3 models"
                            )
                        }
                    }
                )
            }.value

            guard !Task.isCancelled else { return }

            if self.isModelDownloadInProgress {
                self.isModelDownloadInProgress = false
                self.activeModelDownloadPath = nil
                self.activeModelDownloadSessionID = nil
                if result.exitCode == 0 {
                    self.sa3PredownloadProgress = 1
                    self.modelDownloadStatusMessage = "sa3 required models downloaded."
                    self.refreshSA3OfflinePredownloadInventory()
                } else {
                    self.sa3PredownloadProgress = 0
                    self.modelDownloadStatusMessage = result.output.isEmpty
                        ? "sa3 predownload failed."
                        : result.output
                }
            }

            self.sa3LocalDownloadTask = nil
        }
    }

    private func handleOfflineHelperDownloadOutputLine(_ line: String, modelPath: String) {
        guard let data = line.data(using: .utf8),
              let response = try? JSONDecoder().decode(RemotePredownloadStatusResponse.self, from: data)
        else {
            return
        }

        let normalizedProgress = derivedPredownloadProgress(from: response)
        let progress = Double(normalizedProgress) / 100.0
        let queueMessage = response.queueStatus?.message?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let fallbackMessage: String
        switch response.status {
        case "completed":
            fallbackMessage = "downloaded"
        case "failed":
            fallbackMessage = response.error ?? "download failed"
        case "warming", "processing":
            fallbackMessage = "downloading..."
        default:
            fallbackMessage = response.status
        }
        let statusMessage = queueMessage.isEmpty ? fallbackMessage : queueMessage

        if modelPath == "required sa3 models" {
            sa3PredownloadTargetLabel = modelPath
            sa3PredownloadProgress = progress
            modelDownloadStatusMessage = statusMessage
            return
        }

        setModelDownloadState(
            for: modelPath,
            isDownloading: response.status == "warming" || response.status == "processing",
            downloaded: response.status == "completed",
            progress: progress,
            statusMessage: statusMessage
        )
        modelDownloadStatusMessage = statusMessage
    }

    private func resolveCareyDownloadScriptURL(for runtime: ServiceRuntime) -> URL? {
        let runtimeRoot = runtime.service.workingDirectory
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .standardizedFileURL
        let candidates = [
            runtimeRoot.appendingPathComponent("scripts/download_carey_models.sh"),
            runtimeRoot.appendingPathComponent("download_carey_models.sh"),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private func resolveCareyCheckpointDirectory(for runtime: ServiceRuntime) -> URL {
        let configured = runtime.service.environment["ACESTEP_CHECKPOINT_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !configured.isEmpty {
            let expanded = NSString(string: configured).expandingTildeInPath
            if expanded.hasPrefix("/") {
                return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
            }
            return runtime.service.workingDirectory
                .appendingPathComponent(expanded, isDirectory: true)
                .standardizedFileURL
        }
        return runtime.service.workingDirectory
            .appendingPathComponent("checkpoints", isDirectory: true)
            .standardizedFileURL
    }

    private static func resolveCareyModelFileURL(
        baseCheckpointDirectory: URL,
        relativePath: String
    ) -> URL {
        let prefix = "checkpoints/"
        let normalizedRelativePath: String
        if relativePath.hasPrefix(prefix) {
            normalizedRelativePath = String(relativePath.dropFirst(prefix.count))
        } else {
            normalizedRelativePath = relativePath
        }
        return baseCheckpointDirectory
            .appendingPathComponent(normalizedRelativePath)
            .standardizedFileURL
    }

    nonisolated private static func runCareyDownloadScript(
        scriptURL: URL,
        currentDirectory: URL,
        extraEnvironment: [String: String] = [:],
        onOutputLine: (@Sendable (String) -> Void)? = nil
    ) -> (exitCode: Int32, message: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptURL.path]
        process.currentDirectoryURL = currentDirectory
        var environment = ProcessInfo.processInfo.environment
        for (key, value) in extraEnvironment {
            environment[key] = value
        }
        process.environment = environment

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        do {
            try process.run()
            let outputReader = outputPipe.fileHandleForReading
            var lineBuffer = Data()
            var outputLines: [String] = []

            func emitLine(_ rawLine: String) {
                let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !line.isEmpty else { return }
                outputLines.append(line)
                if outputLines.count > 200 {
                    outputLines.removeFirst(outputLines.count - 200)
                }
                onOutputLine?(line)
            }

            while true {
                let chunk = outputReader.availableData
                if chunk.isEmpty {
                    break
                }
                lineBuffer.append(chunk)

                while let newlineIndex = lineBuffer.firstIndex(of: 0x0A) {
                    let lineData = lineBuffer.subdata(in: lineBuffer.startIndex..<newlineIndex)
                    lineBuffer.removeSubrange(lineBuffer.startIndex...newlineIndex)
                    emitLine(String(decoding: lineData, as: UTF8.self))
                }
            }

            if !lineBuffer.isEmpty {
                emitLine(String(decoding: lineBuffer, as: UTF8.self))
            }
            process.waitUntilExit()

            let tailLines = outputLines.suffix(4).joined(separator: " | ")

            if process.terminationStatus == 0 {
                let message = tailLines.isEmpty
                    ? "carey focused download completed."
                    : "carey focused download completed. \(tailLines)"
                return (0, message)
            }

            let message = tailLines.isEmpty
                ? "carey focused download failed (exit \(process.terminationStatus))."
                : "carey focused download failed (exit \(process.terminationStatus)): \(tailLines)"
            return (process.terminationStatus, message)
        } catch {
            return (1, "failed to launch carey focused download: \(error.localizedDescription)")
        }
    }

    private func currentCareyRuntime() -> ServiceRuntime? {
        manager?.services.first(where: { $0.id == "carey" })
    }

    private func currentGaryRuntime() -> ServiceRuntime? {
        manager?.services.first(where: { $0.id == "audiocraft_mlx" })
    }

    private func currentStableAudioRuntime() -> ServiceRuntime? {
        manager?.services.first(where: { $0.id == "stable_audio" })
    }

    private func currentSA3Runtime() -> ServiceRuntime? {
        manager?.services.first(where: { $0.id == "sa3" })
    }

    private func currentFoundationRuntime() -> ServiceRuntime? {
        manager?.services.first(where: { $0.id == "foundation" })
    }

    private func currentMelodyflowRuntime() -> ServiceRuntime? {
        manager?.services.first(where: { $0.id == "melodyflow" })
    }

    private func garyOfflinePredownloadEnvironment(for runtime: ServiceRuntime) -> [String: String] {
        var environment = runtime.service.environment
        environment["HF_HUB_DISABLE_XET"] = "1"
        environment["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        environment["G4L_HF_XET_MODE"] = "off"
        environment["G4L_HF_DOWNLOADER_XET_MODE"] = "off"
        return environment
    }

    private func sa3OfflinePredownloadEnvironment(for runtime: ServiceRuntime) -> [String: String] {
        var environment = runtime.service.environment
        if let token = sharedHuggingFaceToken?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty {
            environment["HF_TOKEN"] = token
            environment["HUGGING_FACE_HUB_TOKEN"] = token
        }
        environment["SA3_PEAK_NORMALIZE_DB"] = sa3PeakNormalizeDb
        environment["SA3_LIMITER_CEILING_DB"] = sa3LimiterCeilingDb
        environment["SA3_LATENT_RESCALE"] = sa3LatentRescale
        environment["SA3_LATENT_SHIFT"] = sa3LatentShift
        environment["SA3_LATENT_TARGET_STD"] = sa3LatentTargetStd
        environment["SA3_CONTINUE_TAIL_PAD"] = sa3ContinuationTailPad
        environment["SA3_MLX_DIT_DTYPE"] = sa3UseFP32DiT ? "float32" : "float16"
        return environment
    }

    private func stableAudioOfflinePredownloadEnvironment(for runtime: ServiceRuntime) -> [String: String] {
        var environment = runtime.service.environment
        if let token = sharedHuggingFaceToken?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty {
            environment["HF_TOKEN"] = token
            environment["HUGGING_FACE_HUB_TOKEN"] = token
        }
        environment["HF_HUB_DISABLE_XET"] = "1"
        environment["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        environment["G4L_HF_XET_MODE"] = "off"
        environment["G4L_HF_DOWNLOADER_XET_MODE"] = "off"
        return environment
    }

    private func reloadSA3AdaptersIfRunning() async throws -> Bool {
        guard let runtime = currentSA3Runtime(), runtime.isRunning else {
            return false
        }

        var components = URLComponents(
            url: runtime.service.healthCheck.url,
            resolvingAgainstBaseURL: false
        )
        components?.path = "/reload"
        components?.query = nil
        components?.fragment = nil
        guard let reloadURL = components?.url else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2212,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Could not resolve the SA3 adapter reload endpoint."
                ]
            )
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 300
        configuration.timeoutIntervalForResource = 360
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }

        for attempt in 0..<150 {
            var request = URLRequest(url: reloadURL)
            request.httpMethod = "POST"
            request.timeoutInterval = 300
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw NSError(
                    domain: "ControlCenterViewModel",
                    code: 2213,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "SA3 returned an invalid adapter reload response."
                    ]
                )
            }
            if (200..<300).contains(http.statusCode) {
                return true
            }
            if http.statusCode == 409, attempt < 149 {
                guard currentSA3Runtime()?.isRunning == true else {
                    throw NSError(
                        domain: "ControlCenterViewModel",
                        code: 2214,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "SA3 stopped while waiting to reload its adapters."
                        ]
                    )
                }
                try await Task.sleep(nanoseconds: 2_000_000_000)
                continue
            }

            let detail = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let detailSuffix: String
            if let detail, !detail.isEmpty {
                detailSuffix = ": \(detail)"
            } else {
                detailSuffix = "."
            }
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2215,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "SA3 adapter reload failed with HTTP \(http.statusCode)"
                        + detailSuffix
                ]
            )
        }

        throw NSError(
            domain: "ControlCenterViewModel",
            code: 2216,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "SA3 remained busy for five minutes; adapter reload timed out."
            ]
        )
    }

    nonisolated private static func renameReplacingNothing(
        source: URL,
        destination: URL
    ) throws {
        guard Darwin.rename(source.path, destination.path) == 0 else {
            let code = errno
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(code),
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Could not move \(source.lastPathComponent) to "
                        + "\(destination.lastPathComponent): "
                        + String(cString: strerror(code))
                ]
            )
        }
    }

    private func sanitizeCareyLoraName(_ raw: String) throws -> String {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2101,
                userInfo: [NSLocalizedDescriptionKey: "LoRA name is required."]
            )
        }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyz0123456789_-")
        guard normalized.unicodeScalars.allSatisfy({ allowed.contains($0) }) else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2102,
                userInfo: [NSLocalizedDescriptionKey: "LoRA name must use lowercase letters, numbers, '-' or '_'."]
            )
        }
        return normalized
    }

    private func sanitizeSA3LoraName(_ raw: String) throws -> String {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2206,
                userInfo: [NSLocalizedDescriptionKey: "LoRA name is required."]
            )
        }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyz0123456789_-")
        guard normalized.unicodeScalars.allSatisfy({ allowed.contains($0) }) else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 2207,
                userInfo: [NSLocalizedDescriptionKey: "LoRA name must use lowercase letters, numbers, '-' or '_'."]
            )
        }
        return normalized
    }

    private func expandedFileURL(from rawPath: String, relativeTo baseURL: URL) -> URL {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let expanded = NSString(string: trimmed).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return URL(fileURLWithPath: expanded, relativeTo: baseURL).standardizedFileURL
    }

    private func expandedFileURL(from rawPath: String) -> URL {
        expandedFileURL(from: rawPath, relativeTo: careyWrapperDirectoryURL())
    }

    private func defaultCareyStorageDirectory() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Application Support/GaryLocalhost/carey", isDirectory: true)
    }

    private func careyWrapperDirectoryURL() -> URL {
        currentCareyRuntime()?.service.workingDirectory.standardizedFileURL
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("ace-lego/wrapper", isDirectory: true)
                .standardizedFileURL
    }

    private func defaultSA3StorageDirectory() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Application Support/GaryLocalhost/sa3", isDirectory: true)
    }

    private func sa3WorkingDirectoryURL() -> URL {
        currentSA3Runtime()?.service.workingDirectory.standardizedFileURL
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("sa3", isDirectory: true)
                .standardizedFileURL
    }

    private func sa3LoraRegistryURL() -> URL {
        if let configured = currentSA3Runtime()?.service.environment["SA3_LORA_REGISTRY"]?.nilIfEmpty {
            return expandedFileURL(from: configured, relativeTo: sa3WorkingDirectoryURL())
        }
        return defaultSA3StorageDirectory().appendingPathComponent("lora_registry.json")
    }

    private func sa3LoraDirectoryURL() -> URL {
        if let configured = currentSA3Runtime()?.service.environment["SA3_LORA_DIR"]?.nilIfEmpty {
            return expandedFileURL(from: configured, relativeTo: sa3WorkingDirectoryURL())
        }
        return defaultSA3StorageDirectory().appendingPathComponent(
            "loras",
            isDirectory: true
        )
    }

    private func sa3PromptsDirectoryURL() -> URL {
        if let configured = currentSA3Runtime()?.service.environment["SA3_PROMPTS_DIR"]?.nilIfEmpty {
            return expandedFileURL(from: configured, relativeTo: sa3WorkingDirectoryURL())
        }
        return defaultSA3StorageDirectory().appendingPathComponent("prompts", isDirectory: true)
    }

    private func sa3LoraCatalogURL() -> URL {
        sa3LoraRegistryURL().deletingLastPathComponent().appendingPathComponent("lora_catalog.json")
    }

    private func sa3DefaultPromptsURL() -> URL {
        sa3WorkingDirectoryURL().appendingPathComponent("prompts/defaults.json")
    }

    private func sa3PythonExecutableURL() -> URL? {
        if let venvDirectory = currentSA3Runtime()?.service.bootstrap?.venvDirectory {
            let url = venvDirectory.appendingPathComponent("bin/python")
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }

    private func careyLoraRegistryURL() -> URL {
        if let configured = currentCareyRuntime()?.service.environment["CAREY_LORA_REGISTRY"]?.nilIfEmpty {
            return expandedFileURL(from: configured)
        }
        return defaultCareyStorageDirectory().appendingPathComponent("lora_registry.json")
    }

    private func careyCaptionsURL() -> URL {
        if let configured = currentCareyRuntime()?.service.environment["CAREY_CAPTIONS"]?.nilIfEmpty {
            return expandedFileURL(from: configured)
        }
        return defaultCareyStorageDirectory().appendingPathComponent("captions.json")
    }

    private func careyCheckpointDirectoryURL() -> URL {
        if let configured = currentCareyRuntime()?.service.environment["ACESTEP_CHECKPOINT_DIR"]?.nilIfEmpty {
            return expandedFileURL(from: configured)
        }
        if let configured = currentCareyRuntime()?.service.environment["CAREY_CHECKPOINT_DIR"]?.nilIfEmpty {
            return expandedFileURL(from: configured)
        }
        return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(
                "Library/Application Support/GaryLocalhost/cache/carey/checkpoints",
                isDirectory: true
            )
    }

    private func careyLoraCatalogURL() -> URL {
        careyLoraRegistryURL().deletingLastPathComponent().appendingPathComponent("lora_catalog.json")
    }

    private func careyDefaultCaptionsURL() -> URL {
        careyWrapperDirectoryURL().appendingPathComponent("default_captions.json")
    }

    private func careyPythonExecutableURL() -> URL? {
        if let configured = currentCareyRuntime()?.service.environment["CAREY_PYTHON"]?.nilIfEmpty {
            let url = expandedFileURL(from: configured)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        if let venvDirectory = currentCareyRuntime()?.service.bootstrap?.venvDirectory {
            let url = venvDirectory.appendingPathComponent("bin/python")
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }

    private static func normalizeCareyLoraBackends(_ values: [String]) -> [String] {
        var cleaned: [String] = []
        for value in values {
            let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if ["base", "turbo", "regular"].contains(normalized), !cleaned.contains(normalized) {
                cleaned.append(normalized)
            }
        }
        return cleaned.isEmpty ? ["base", "turbo"] : cleaned
    }

    private static func normalizeCareyLoraModelFamily(_ raw: String) -> String {
        raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "xl" ? "xl" : "standard"
    }

    private static func inferCareyLoraModelFamily(from checkpointDirectory: URL) -> String {
        checkpointDirectory.lastPathComponent.lowercased().contains("xl") ? "xl" : "standard"
    }

    private static func looksLikeCareyLoraCheckpointDirectory(_ directory: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return false
        }

        let requiredNames = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "pytorch_lora_weights.safetensors",
        ]
        if requiredNames.contains(where: { FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }) {
            return true
        }

        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }

        return contents.contains { $0.pathExtension.lowercased() == "safetensors" }
    }

    private static func countCareyCaptionSidecars(in directory: URL) -> Int {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return 0
        }
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else {
            return 0
        }
        return contents.filter { url in
            guard url.pathExtension.lowercased() == "txt" else { return false }
            let name = url.lastPathComponent
            return !name.contains(".v4bak") && !name.hasSuffix(".v4bak")
        }.count
    }

    private static func looksLikeSA3LoraCheckpoint(_ file: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            return false
        }
        let ext = file.pathExtension.lowercased()
        return ext == "ckpt" || ext == "safetensors"
    }

    private static func countSA3CaptionSidecars(in directory: URL) -> Int {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return 0
        }
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else {
            return 0
        }
        return contents.filter { url in
            guard url.pathExtension.lowercased() == "txt" else { return false }
            let name = url.lastPathComponent
            return !name.contains(".v4bak") && !name.hasSuffix(".v4bak")
        }.count
    }

    private func sa3PromptFileURL(for name: String) -> URL {
        sa3PromptsDirectoryURL().appendingPathComponent("\(name).json")
    }

    private func readSA3LoraCatalog(at url: URL) throws -> [String: SA3LoraCatalogEntry] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return [:]
        }

        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String: SA3LoraCatalogEntry].self, from: data)
        var normalized: [String: SA3LoraCatalogEntry] = [:]
        for (name, var entry) in decoded {
            let trimmedPath = entry.path.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedPath.isEmpty else { continue }
            entry.path = trimmedPath
            entry.promptsPath = entry.promptsPath?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
            entry.trainingBaseModel = entry.trainingBaseModel?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .nilIfEmpty
            entry.inferenceModel = entry.inferenceModel?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .nilIfEmpty
            entry.trainingJobId = entry.trainingJobId?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .nilIfEmpty
            if !entry.strength.isFinite {
                entry.strength = 1.0
            }
            var checkpointsByStep: [Int: SA3TrainingCheckpoint] = [:]
            for var checkpoint in entry.trainingCheckpoints where checkpoint.step >= 0 {
                checkpoint.path = checkpoint.path
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !checkpoint.path.isEmpty else { continue }
                checkpointsByStep[checkpoint.step] = checkpoint
            }
            entry.trainingCheckpoints = checkpointsByStep.values.sorted {
                if $0.step != $1.step { return $0.step < $1.step }
                if $0.epoch != $1.epoch { return $0.epoch < $1.epoch }
                return $0.path < $1.path
            }
            if let selectedStep = entry.selectedTrainingStep,
               !entry.trainingCheckpoints.contains(where: { $0.step == selectedStep }) {
                entry.selectedTrainingStep = nil
            }
            entry = inferSA3TrainingHistory(
                for: name,
                catalogEntry: entry
            )
            normalized[name] = entry
        }
        return normalized
    }

    private func inferSA3TrainingHistory(
        for name: String,
        catalogEntry: SA3LoraCatalogEntry
    ) -> SA3LoraCatalogEntry {
        guard catalogEntry.trainingCheckpoints.isEmpty else {
            return catalogEntry
        }

        let activeURL = expandedFileURL(
            from: catalogEntry.path,
            relativeTo: sa3WorkingDirectoryURL()
        )
        let managedURL = sa3LoraDirectoryURL()
            .appendingPathComponent("\(name).safetensors")
            .standardizedFileURL
        guard activeURL.standardizedFileURL.path == managedURL.path else {
            return catalogEntry
        }

        let jobsURL = defaultSA3StorageDirectory()
            .appendingPathComponent("training/jobs", isDirectory: true)
        guard let jobDirectories = try? FileManager.default.contentsOfDirectory(
            at: jobsURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return catalogEntry
        }

        for jobURL in jobDirectories.sorted(
            by: { $0.lastPathComponent > $1.lastPathComponent }
        ) {
            let statusURL = jobURL.appendingPathComponent("status.json")
            guard let data = try? Data(contentsOf: statusURL),
                  let status = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                  status["status"] as? String == "completed",
                  status["name"] as? String == name,
                  let jobID = status["job_id"] as? String,
                  let maxSteps = status["max_steps"] as? Int else {
                continue
            }

            var checkpointsByStep: [Int: SA3TrainingCheckpoint] = [:]
            if let files = try? FileManager.default.contentsOfDirectory(
                at: jobURL,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ) {
                for fileURL in files {
                    let filename = fileURL.lastPathComponent
                    let prefix = "gary-mlx-lora-step-"
                    let suffix = ".safetensors"
                    guard filename.hasPrefix(prefix),
                          filename.hasSuffix(suffix) else {
                        continue
                    }
                    let start = filename.index(
                        filename.startIndex,
                        offsetBy: prefix.count
                    )
                    let end = filename.index(
                        filename.endIndex,
                        offsetBy: -suffix.count
                    )
                    guard let step = Int(filename[start..<end]),
                          Self.looksLikeSA3LoraCheckpoint(fileURL) else {
                        continue
                    }
                    checkpointsByStep[step] = SA3TrainingCheckpoint(
                        step: step,
                        path: fileURL.path
                    )
                }
            }

            let finalURL = jobURL.appendingPathComponent(
                "gary-mlx-lora-final.safetensors"
            )
            if Self.looksLikeSA3LoraCheckpoint(finalURL) {
                checkpointsByStep[maxSteps] = SA3TrainingCheckpoint(
                    step: maxSteps,
                    path: finalURL.path
                )
            }
            guard !checkpointsByStep.isEmpty else { continue }

            var inferred = catalogEntry
            inferred.trainingJobId = jobID
            inferred.trainingCheckpoints = checkpointsByStep.values.sorted {
                $0.step < $1.step
            }
            inferred.selectedTrainingStep = maxSteps
            return inferred
        }

        return catalogEntry
    }

    private func saveSA3LoraCatalog(_ catalog: [String: SA3LoraCatalogEntry], to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(catalog)
        try data.write(to: url, options: .atomic)
    }

    private func resolveSA3PromptsSource(for entry: SA3LoraCatalogEntry) -> URL? {
        if let promptsPath = entry.promptsPath?.nilIfEmpty {
            let url = expandedFileURL(from: promptsPath, relativeTo: sa3WorkingDirectoryURL())
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return url
            }
        }

        let checkpointFile = expandedFileURL(from: entry.path, relativeTo: sa3WorkingDirectoryURL())
        let parent = checkpointFile.deletingLastPathComponent()
        return Self.countSA3CaptionSidecars(in: parent) > 0 ? parent : nil
    }

    private func readSA3PromptCount(from url: URL) -> Int {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let dice = json["dice"] as? [String: Any] else {
            return 0
        }
        return dice.values.reduce(0) { partialResult, value in
            partialResult + ((value as? [Any])?.count ?? 0)
        }
    }

    private func readSA3PromptPools() -> [String: Int] {
        var pools: [String: Int] = [:]
        let directory = sa3PromptsDirectoryURL()
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else {
            return pools
        }

        for fileURL in contents where fileURL.pathExtension.lowercased() == "json" {
            let name = fileURL.deletingPathExtension().lastPathComponent
            pools[name] = readSA3PromptCount(from: fileURL)
        }
        return pools
    }

    private func buildSA3LoraEntries(from catalog: [String: SA3LoraCatalogEntry]) -> [SA3LoraEntry] {
        catalog.keys.sorted().compactMap { name in
            guard let entry = catalog[name] else { return nil }
            let checkpointFile = expandedFileURL(from: entry.path, relativeTo: sa3WorkingDirectoryURL())
            let checkpointExists = Self.looksLikeSA3LoraCheckpoint(checkpointFile)
            let resolvedPromptsURL = resolveSA3PromptsSource(for: entry)
            let captionCount = resolvedPromptsURL.map { Self.countSA3CaptionSidecars(in: $0) } ?? 0
            let promptFile = sa3PromptFileURL(for: name)
            let promptFileExists = FileManager.default.fileExists(atPath: promptFile.path)
            let promptCount = readSA3PromptCount(from: promptFile)
            let trainingCheckpoints = entry.trainingCheckpoints.compactMap {
                checkpoint -> SA3TrainingCheckpoint? in
                let checkpointURL = expandedFileURL(
                    from: checkpoint.path,
                    relativeTo: sa3WorkingDirectoryURL()
                )
                guard Self.looksLikeSA3LoraCheckpoint(checkpointURL) else {
                    return nil
                }
                return SA3TrainingCheckpoint(
                    step: checkpoint.step,
                    epoch: checkpoint.epoch,
                    path: checkpointURL.path
                )
            }
            let selectedTrainingStep = entry.selectedTrainingStep.flatMap { step in
                trainingCheckpoints.contains(where: { $0.step == step }) ? step : nil
            }

            return SA3LoraEntry(
                name: name,
                path: checkpointFile.path,
                promptsPath: entry.promptsPath,
                resolvedPromptsPath: resolvedPromptsURL?.path,
                promptFilePath: promptFile.path,
                promptFileExists: promptFileExists,
                promptCount: promptCount,
                captionCount: captionCount,
                strength: entry.strength,
                checkpointExists: checkpointExists,
                registered: checkpointExists,
                trainingJobId: entry.trainingJobId,
                trainingCheckpoints: trainingCheckpoints,
                selectedTrainingStep: selectedTrainingStep
            )
        }
    }

    private func writeSA3LoraRegistry(entries: [SA3LoraEntry], to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )

        var payload: [String: [String: Any]] = [:]
        for entry in entries where entry.registered {
            payload[entry.name] = [
                "path": entry.path,
                "strength": entry.strength,
            ]
        }

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private func ensureDefaultSA3Prompts() throws {
        let source = sa3DefaultPromptsURL()
        guard FileManager.default.fileExists(atPath: source.path) else { return }

        let destination = sa3PromptsDirectoryURL().appendingPathComponent("defaults.json")
        guard !FileManager.default.fileExists(atPath: destination.path) else { return }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        try FileManager.default.copyItem(at: source, to: destination)
    }

    private func buildSA3LoraState() throws -> SA3LoraState {
        try ensureDefaultSA3Prompts()
        let catalogURL = sa3LoraCatalogURL()
        let registryURL = sa3LoraRegistryURL()
        let promptsURL = sa3PromptsDirectoryURL()
        let catalog = try readSA3LoraCatalog(at: catalogURL)
        let entries = buildSA3LoraEntries(from: catalog)
        try writeSA3LoraRegistry(entries: entries, to: registryURL)
        return SA3LoraState(
            entries: entries,
            pools: readSA3PromptPools(),
            catalogPath: catalogURL.path,
            registryPath: registryURL.path,
            promptsDir: promptsURL.path
        )
    }

    private func readCareyLoraCatalog(at url: URL) throws -> [String: CareyLoraCatalogEntry] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return [:]
        }

        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String: CareyLoraCatalogEntry].self, from: data)
        var normalized: [String: CareyLoraCatalogEntry] = [:]
        for (name, var entry) in decoded {
            let trimmedPath = entry.path.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedPath.isEmpty else { continue }
            entry.path = trimmedPath
            entry.captionsPath = entry.captionsPath?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
            entry.scale = entry.scale.isFinite ? entry.scale : 1.0
            entry.backends = Self.normalizeCareyLoraBackends(entry.backends)
            entry.modelFamily = Self.normalizeCareyLoraModelFamily(entry.modelFamily)
            normalized[name] = entry
        }
        return normalized
    }

    private func saveCareyLoraCatalog(_ catalog: [String: CareyLoraCatalogEntry], to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(catalog)
        try data.write(to: url, options: .atomic)
    }

    private func resolveCareyCaptionsSource(for entry: CareyLoraCatalogEntry) -> URL? {
        if let captionsPath = entry.captionsPath?.nilIfEmpty {
            let url = expandedFileURL(from: captionsPath)
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return url
            }
        }

        let checkpointDirectory = expandedFileURL(from: entry.path)
        return Self.countCareyCaptionSidecars(in: checkpointDirectory) > 0 ? checkpointDirectory : nil
    }

    private func loadCareyLoraMetadata(
        checkpointDirectory: URL,
        captionsDirectory: URL?,
        existing: CareyLoraCatalogEntry?
    ) throws -> (Double, [String], String) {
        var candidates = [checkpointDirectory.appendingPathComponent("metadata.json")]
        if let captionsDirectory {
            let candidate = captionsDirectory.appendingPathComponent("metadata.json")
            if !candidates.contains(candidate) {
                candidates.append(candidate)
            }
        }

        for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
            let data = try Data(contentsOf: candidate)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            let scale = (json["scale"] as? Double) ?? 1.0
            let backends = Self.normalizeCareyLoraBackends(json["backends"] as? [String] ?? [])
            let modelFamily = Self.normalizeCareyLoraModelFamily(
                (json["model_family"] as? String)
                ?? (json["family"] as? String)
                ?? Self.inferCareyLoraModelFamily(from: checkpointDirectory)
            )
            return (scale, backends, modelFamily)
        }

        if let existing {
            return (
                existing.scale,
                Self.normalizeCareyLoraBackends(existing.backends),
                Self.normalizeCareyLoraModelFamily(existing.modelFamily)
            )
        }

        return (
            1.0,
            ["base", "turbo"],
            Self.inferCareyLoraModelFamily(from: checkpointDirectory)
        )
    }

    private func buildCareyLoraEntries(from catalog: [String: CareyLoraCatalogEntry]) -> [CareyLoraEntry] {
        catalog.keys.sorted().compactMap { name in
            guard let entry = catalog[name] else { return nil }
            let checkpointURL = expandedFileURL(from: entry.path)
            let checkpointExists = Self.looksLikeCareyLoraCheckpointDirectory(checkpointURL)
            let resolvedCaptionsURL = resolveCareyCaptionsSource(for: entry)
            let captionCount = resolvedCaptionsURL.map { Self.countCareyCaptionSidecars(in: $0) } ?? 0

            return CareyLoraEntry(
                name: name,
                path: checkpointURL.path,
                captionsPath: entry.captionsPath,
                resolvedCaptionsPath: resolvedCaptionsURL?.path,
                captionCount: captionCount,
                scale: entry.scale,
                backends: Self.normalizeCareyLoraBackends(entry.backends),
                modelFamily: Self.normalizeCareyLoraModelFamily(entry.modelFamily),
                checkpointExists: checkpointExists,
                registered: checkpointExists
            )
        }
    }

    private func writeCareyLoraRegistry(entries: [CareyLoraEntry], to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )

        var payload: [String: [String: Any]] = [:]
        for entry in entries where entry.registered {
            payload[entry.name] = [
                "path": entry.path,
                "scale": entry.scale,
                "backends": entry.backends,
                "model_family": entry.modelFamily,
            ]
        }

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private func readCareyCaptionPools(from url: URL) -> [String: Int] {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }

        var pools: [String: Int] = [:]
        for (name, value) in json {
            pools[name] = (value as? [Any])?.count ?? 0
        }
        return pools
    }

    private func readBundledDefaultCareyCaptionPool() -> [String] {
        let url = careyDefaultCaptionsURL()
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["default"] as? [String] else {
            return []
        }
        return items.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }

    private func ensureDefaultCareyCaptions() throws {
        let defaultPool = readBundledDefaultCareyCaptionPool()
        guard !defaultPool.isEmpty else { return }

        let captionsURL = careyCaptionsURL()
        try FileManager.default.createDirectory(
            at: captionsURL.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )

        var payload: [String: Any] = [:]
        if let existingData = try? Data(contentsOf: captionsURL),
           let existingJSON = try? JSONSerialization.jsonObject(with: existingData) as? [String: Any] {
            payload = existingJSON
        }
        payload["default"] = defaultPool
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: captionsURL, options: .atomic)
    }

    private func buildCareyLoraState() throws -> CareyLoraState {
        try ensureDefaultCareyCaptions()
        let catalogURL = careyLoraCatalogURL()
        let registryURL = careyLoraRegistryURL()
        let captionsURL = careyCaptionsURL()
        let catalog = try readCareyLoraCatalog(at: catalogURL)
        let entries = buildCareyLoraEntries(from: catalog)
        try writeCareyLoraRegistry(entries: entries, to: registryURL)
        return CareyLoraState(
            entries: entries,
            pools: readCareyCaptionPools(from: captionsURL),
            catalogPath: catalogURL.path,
            registryPath: registryURL.path,
            captionsPath: captionsURL.path
        )
    }

    private func tryReloadCareyAdminIfRunning() async -> Bool {
        guard let baseURL = modelDownloadAPIBaseURL(for: "carey") else {
            return false
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("admin/reload"))
        request.httpMethod = "POST"
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return false
            }
            return (200...299).contains(http.statusCode)
        } catch {
            return false
        }
    }

    nonisolated private static func runLocalProcess(
        executableURL: URL,
        arguments: [String],
        currentDirectory: URL? = nil,
        environment: [String: String] = [:]
    ) -> (exitCode: Int32, output: String) {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory

        var inheritedEnvironment = ProcessInfo.processInfo.environment
        for (key, value) in environment {
            inheritedEnvironment[key] = value
        }
        process.environment = inheritedEnvironment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let output = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            return (process.terminationStatus, output)
        } catch {
            return (1, "failed to launch \(executableURL.lastPathComponent): \(error.localizedDescription)")
        }
    }

    nonisolated private static func loadLocalPredownloadCatalogOutputs(
        executableURL: URL,
        helperURL: URL,
        currentDirectory: URL,
        environment: [String: String]
    ) -> Result<(catalogOutput: String, statusOutput: String), Error> {
        let catalogResult = runLocalProcess(
            executableURL: executableURL,
            arguments: [helperURL.path, "catalog"],
            currentDirectory: currentDirectory,
            environment: environment
        )
        guard catalogResult.exitCode == 0 else {
            return .failure(
                NSError(
                    domain: "ControlCenterViewModel",
                    code: 5001,
                    userInfo: [NSLocalizedDescriptionKey: catalogResult.output]
                )
            )
        }

        let statusResult = runLocalProcess(
            executableURL: executableURL,
            arguments: [helperURL.path, "status"],
            currentDirectory: currentDirectory,
            environment: environment
        )
        guard statusResult.exitCode == 0 else {
            return .failure(
                NSError(
                    domain: "ControlCenterViewModel",
                    code: 5002,
                    userInfo: [NSLocalizedDescriptionKey: statusResult.output]
                )
            )
        }
        return .success((catalogOutput: catalogResult.output, statusOutput: statusResult.output))
    }

    private func applyOfflinePredownloadCatalogResult(
        _ result: Result<(catalogOutput: String, statusOutput: String), Error>,
        expectedHelperName: String
    ) {
        switch result {
        case .success(let outputs):
            do {
                let decoder = JSONDecoder()
                let catalog = try decoder.decode(
                    RemoteModelsResponse.self,
                    from: Self.jsonDataFromProcessOutput(outputs.catalogOutput)
                )
                let statuses = try decoder.decode(
                    RemoteDownloadStatusResponse.self,
                    from: Self.jsonDataFromProcessOutput(outputs.statusOutput)
                )

                var models = flattenRemoteModels(catalog.models)
                for index in models.indices {
                    if let status = statuses.models[models[index].path] {
                        models[index].downloaded = status.downloaded
                        if status.downloaded {
                            models[index].statusMessage = "downloaded"
                        } else if let missing = status.missing, !missing.isEmpty {
                            models[index].statusMessage = "missing \(missing.count) dependency\(missing.count == 1 ? "" : "ies")"
                        } else {
                            models[index].statusMessage = "not downloaded"
                        }
                    } else {
                        models[index].statusMessage = "unknown"
                    }
                }
                downloadableModels = models
                modelDownloadStatusMessage = "pick a model to pre-download for offline usage."
            } catch {
                downloadableModels = []
                modelDownloadStatusMessage = "failed to parse \(expectedHelperName) output: \(error.localizedDescription)"
            }
        case .failure(let error):
            downloadableModels = []
            modelDownloadStatusMessage = error.localizedDescription
        }

        isModelCatalogLoading = false
    }

    nonisolated private static func runStreamingLocalProcess(
        executableURL: URL,
        arguments: [String],
        currentDirectory: URL? = nil,
        environment: [String: String] = [:],
        onOutputLine: (@Sendable (String) -> Void)? = nil
    ) -> (exitCode: Int32, output: String) {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory

        var inheritedEnvironment = ProcessInfo.processInfo.environment
        for (key, value) in environment {
            inheritedEnvironment[key] = value
        }
        process.environment = inheritedEnvironment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            let outputReader = pipe.fileHandleForReading
            var lineBuffer = Data()
            var outputLines: [String] = []

            func emitLine(_ rawLine: String) {
                let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !line.isEmpty else { return }
                outputLines.append(line)
                if outputLines.count > 200 {
                    outputLines.removeFirst(outputLines.count - 200)
                }
                onOutputLine?(line)
            }

            while true {
                let chunk = outputReader.availableData
                if chunk.isEmpty {
                    break
                }
                lineBuffer.append(chunk)

                while let newlineIndex = lineBuffer.firstIndex(of: 0x0A) {
                    let lineData = lineBuffer.subdata(in: lineBuffer.startIndex..<newlineIndex)
                    lineBuffer.removeSubrange(lineBuffer.startIndex...newlineIndex)
                    emitLine(String(decoding: lineData, as: UTF8.self))
                }
            }

            if !lineBuffer.isEmpty {
                emitLine(String(decoding: lineBuffer, as: UTF8.self))
            }

            process.waitUntilExit()
            return (process.terminationStatus, outputLines.suffix(4).joined(separator: " | "))
        } catch {
            return (1, "failed to launch \(executableURL.lastPathComponent): \(error.localizedDescription)")
        }
    }

    nonisolated private static func jsonDataFromProcessOutput(_ output: String) throws -> Data {
        let jsonLine = output
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .last { !$0.isEmpty }

        guard let jsonLine, !jsonLine.isEmpty else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 5003,
                userInfo: [NSLocalizedDescriptionKey: "expected JSON output from foundation helper."]
            )
        }

        return Data(jsonLine.utf8)
    }

    nonisolated private static func userFacingStableAudioPredownloadFailureMessage(from output: String) -> String {
        let filteredLines = output
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { line in
                !line.isEmpty
                && !line.localizedCaseInsensitiveContains("Riffs directory not found")
                && !line.localizedCaseInsensitiveContains("pkg_resources is deprecated as an API")
            }

        let filteredOutput = filteredLines.joined(separator: "\n")
        if let response = stableAudioPredownloadStatusFromProcessOutput(filteredOutput) {
            let queueMessage = response.queueMessage?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let errorMessage = response.error?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let candidate: String
            if !errorMessage.isEmpty {
                candidate = errorMessage
            } else if !queueMessage.isEmpty {
                candidate = queueMessage
            } else {
                candidate = filteredOutput.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return sanitizeStableAudioPredownloadMessage(candidate)
        }

        return sanitizeStableAudioPredownloadMessage(
            filteredOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    nonisolated private static func stableAudioPredownloadStatusFromProcessOutput(
        _ output: String
    ) -> (error: String?, queueMessage: String?)? {
        let candidates = output
            .components(separatedBy: "\n")
            .flatMap { line in line.components(separatedBy: " | ") }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        for candidate in candidates.reversed() {
            guard candidate.first == "{", candidate.last == "}" else { continue }
            guard let data = candidate.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                continue
            }

            let error = json["error"] as? String
            let queueStatus = json["queue_status"] as? [String: Any]
            let queueMessage = queueStatus?["message"] as? String
            if error != nil || queueMessage != nil {
                return (error: error, queueMessage: queueMessage)
            }
        }
        return nil
    }

    nonisolated private static func sanitizeStableAudioPredownloadMessage(_ message: String?) -> String {
        let trimmed = message?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else {
            return "stable audio download failed."
        }

        let lower = trimmed.lowercased()
        if lower.contains("hf_token is required") {
            return "save your hugging face token in jerry setup first."
        }
        if lower.contains("invalid user token")
            || lower.contains("token from hf_token environment variable is invalid")
            || lower.contains("authorization header")
            || lower.contains("invalid credentials")
            || (lower.contains("401") && lower.contains("token"))
            || (lower.contains("hf_token") && lower.contains("invalid"))
            || (lower.contains("hugging face") && lower.contains("token") && lower.contains("invalid")) {
            return "saved hugging face token was rejected. Open token settings, paste a fresh token, and try again."
        }

        return trimmed
    }

    private func modelDownloadAPIBaseURL(for serviceID: String) -> URL? {
        guard let runtime = manager?.services.first(where: { $0.id == serviceID }) else {
            return nil
        }
        guard runtime.processState == .running else {
            return nil
        }
        guard var components = URLComponents(
            url: runtime.service.healthCheck.url,
            resolvingAgainstBaseURL: false
        ) else {
            return nil
        }
        components.path = ""
        components.query = nil
        components.fragment = nil
        return components.url
    }

    private func modelDownloadDisplayName(forServiceID serviceID: String) -> String {
        switch serviceID {
        case "audiocraft_mlx":
            return "gary (musicgen)"
        case "melodyflow":
            return "terry (melodyflow)"
        case "sa3":
            return "sa3 (stable audio 3)"
        case "stable_audio":
            return "jerry (stable audio)"
        case "carey":
            return "carey (ace lego)"
        case "foundation":
            return "foundation-1"
        default:
            if let runtime = manager?.services.first(where: { $0.id == serviceID }) {
                return runtime.service.name
            }
            return serviceID
        }
    }

    private func foundationPredownloadHelperURL(for runtime: ServiceRuntime) -> URL? {
        let helperURL = runtime.service.workingDirectory
            .appendingPathComponent("foundation_predownload_cli.py")
            .standardizedFileURL
        return FileManager.default.fileExists(atPath: helperURL.path) ? helperURL : nil
    }

    private func garyPredownloadHelperURL(for runtime: ServiceRuntime) -> URL? {
        let helperURL = runtime.service.workingDirectory
            .appendingPathComponent("g4l_predownload_cli.py")
            .standardizedFileURL
        return FileManager.default.fileExists(atPath: helperURL.path) ? helperURL : nil
    }

    private func melodyflowPredownloadHelperURL(for runtime: ServiceRuntime) -> URL? {
        let helperURL = runtime.service.workingDirectory
            .appendingPathComponent("melodyflow_predownload_cli.py")
            .standardizedFileURL
        return FileManager.default.fileExists(atPath: helperURL.path) ? helperURL : nil
    }

    private func stableAudioPredownloadHelperURL(for runtime: ServiceRuntime) -> URL? {
        let helperURL = runtime.service.workingDirectory
            .appendingPathComponent("stable_predownload_cli.py")
            .standardizedFileURL
        return FileManager.default.fileExists(atPath: helperURL.path) ? helperURL : nil
    }

    private func sa3PredownloadHelperURL(for runtime: ServiceRuntime) -> URL? {
        let helperURL = runtime.service.workingDirectory
            .appendingPathComponent("sa3_predownload_cli.py")
            .standardizedFileURL
        return FileManager.default.fileExists(atPath: helperURL.path) ? helperURL : nil
    }

    private func ensureHTTP200(response: URLResponse, body: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw NSError(
                domain: "ControlCenterViewModel",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "unexpected response from backend."]
            )
        }
        guard (200...299).contains(http.statusCode) else {
            if let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
               let message = json["error"] as? String,
               !message.isEmpty {
                throw NSError(domain: "ControlCenterViewModel", code: http.statusCode, userInfo: [
                    NSLocalizedDescriptionKey: message
                ])
            }
            throw NSError(domain: "ControlCenterViewModel", code: http.statusCode, userInfo: [
                NSLocalizedDescriptionKey: "backend returned http \(http.statusCode)."
            ])
        }
    }

    private func applyStableAudioTokenState(configured: Bool) {
        stableAudioTokenConfigured = configured
        if configured {
            if stableAudioTokenStatus.isEmpty {
                stableAudioTokenStatus = "token already saved in keychain."
            }
        } else if stableAudioTokenStatus == "token already saved in keychain." {
            stableAudioTokenStatus = ""
        }
    }

    private func observeApplicationTermination() {
        NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                self?.logRefreshTask?.cancel()
                self?.modelDownloadPollTask?.cancel()
                self?.foundationLocalDownloadTask?.cancel()
                self?.garyLocalDownloadTask?.cancel()
                self?.melodyflowLocalDownloadTask?.cancel()
                self?.stableAudioLocalDownloadTask?.cancel()
                self?.sa3LocalDownloadTask?.cancel()
                self?.careyDownloadTask?.cancel()
                self?.manager?.shutdownForApplicationTermination()
            }
            .store(in: &cancellables)
    }
}
