import Foundation
import SwiftUI
import Combine
import AppKit

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
        }
    }
}

@MainActor
final class ControlCenterViewModel: ObservableObject {
    private static let stableAudioBackendDefaultsKey = "stableAudioBackendEngine"
    private static let melodyFlowBackendDefaultsKey = "melodyFlowBackendEngine"
    private static let careyBackendDefaultsKey = "careyBackendEngine"
    private static let careyUseXlModelsDefaultsKey = "careyUseXlModels"
    private static let careyUseSampledMlxVaeEncodeDefaultsKey = "careyUseSampledMlxVaeEncode"
    private static let experimentalCareyMlxVaeEncodeToggleDefaultsKey = "experimentalCareyMlxVaeEncodeToggleEnabled"

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
    @Published var careyRequiredModels: [CareyRequiredModelStatus] = []
    @Published var isCareyDownloadInProgress: Bool = false
    @Published var isCareyLifecycleActionInProgress: Bool = false
    @Published var careyPredownloadProgress: Double = 0
    @Published var careyPredownloadActiveLabel: String = ""
    @Published var stableAudioBackendEngine: StableAudioBackendEngine = .mps
    @Published var melodyFlowBackendEngine: MelodyFlowBackendEngine = .mps
    @Published var careyBackendEngine: CareyBackendEngine = .mlx
    @Published var careyUseXlModels: Bool = false
    @Published var careyUseSampledMlxVaeEncode: Bool = true
    @Published var melodyFlowBackendStatus: String = ""
    @Published var careyBackendStatus: String = ""
    @Published var rebuildFailureReport: RebuildFailureReport?
    @Published var rebuildDiagnosticsStatusMessage: String = ""
    @Published var isRequirementsEditorPresented: Bool = false
    @Published var requirementsEditorPath: String = ""
    @Published var requirementsEditorText: String = ""
    @Published var requirementsEditorStatusMessage: String = ""
    @Published var modelDownloadServiceID: String = "audiocraft_mlx"

    private var logRefreshTask: Task<Void, Never>?
    private var modelDownloadPollTask: Task<Void, Never>?
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
        if UserDefaults.standard.object(forKey: Self.careyUseSampledMlxVaeEncodeDefaultsKey) != nil {
            careyUseSampledMlxVaeEncode = UserDefaults.standard.bool(
                forKey: Self.careyUseSampledMlxVaeEncodeDefaultsKey
            )
        } else {
            careyUseSampledMlxVaeEncode = true
        }
        observeApplicationTermination()
        refreshStableAudioTokenState()
        loadManifest()
    }

    deinit {
        logRefreshTask?.cancel()
        modelDownloadPollTask?.cancel()
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
        guard let runtime = manager?.services.first(where: { $0.id == modelDownloadServiceID }) else {
            return false
        }
        return runtime.processState == .running
    }

    var modelDownloadServiceDisplayName: String {
        modelDownloadDisplayName(forServiceID: modelDownloadServiceID)
    }

    var canManageStableAudioPredownloads: Bool {
        guard modelDownloadServiceID == "stable_audio" else { return false }
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

    private func activeCareyConfigNames() -> (base: String, sft: String, turbo: String) {
        if careyUseXlModels {
            return ("acestep-v15-xl-base", "acestep-v15-xl-sft", "acestep-v15-xl-turbo")
        }
        return ("acestep-v15-base", "acestep-v15-sft", "acestep-v15-turbo")
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
        careyDownloadTask?.cancel()
        careyDownloadTask = nil
        isModelDownloadInProgress = false
        isCareyDownloadInProgress = false
        isCareyLifecycleActionInProgress = false
        activeModelDownloadPath = nil
        activeModelDownloadSessionID = nil
        careyActiveDownloadTargets = []
        careyRequiredModels = []
        rebuildFailureReport = nil
        rebuildDiagnosticsStatusMessage = ""
        managerCancellables.removeAll()

        let defaultURL = ManifestLoader.defaultManifestURL()
        manifestPath = defaultURL.path
        reloadHFScreenshots()
        refreshStableAudioTokenState()

        do {
            let manifest = try ManifestLoader.load(from: defaultURL)
            let manager = ServiceManager(manifest: manifest)
            manager.setStableAudioBackendEngine(stableAudioBackendEngine.rawValue, restartIfRunning: false)
            manager.setMelodyFlowBackendEngine(melodyFlowBackendEngine.rawValue, restartIfRunning: false)
            manager.setCareyBackendEngine(careyBackendEngine.rawValue, restartIfRunning: false)
            manager.setCareyUseXlModels(careyUseXlModels, restartIfRunning: false)
            manager.setCareyUseSampledMlxVaeEncode(careyUseSampledMlxVaeEncode, restartIfRunning: false)
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
        modelDownloadServiceID = serviceID
        isModelDownloadSheetPresented = true
        if serviceID == "stable_audio" {
            prepareStableAudioPredownloadState()
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

        if modelDownloadServiceID == "carey" {
            prepareCareyPredownloadState()
            return
        }

        if isModelDownloadInProgress {
            modelDownloadStatusMessage = "a model download is already in progress."
            return
        }

        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        activeModelDownloadSessionID = nil
        activeModelDownloadPath = nil
        isModelDownloadInProgress = false

        guard let baseURL = modelDownloadAPIBaseURL(for: modelDownloadServiceID) else {
            downloadableModels = []
            modelDownloadStatusMessage = "start \(modelDownloadServiceDisplayName) to manage model downloads."
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
            modelDownloadStatusMessage = "start \(modelDownloadServiceDisplayName) to fetch checkpoints."
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
        stableAudioPredownloadProgress = 0
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

    func refreshStableAudioPredownloadInventory(checkpointsHint: [String] = []) {
        let serviceID = modelDownloadServiceID
        guard serviceID == "stable_audio" else { return }
        guard let baseURL = modelDownloadAPIBaseURL(for: serviceID) else {
            stableAudioInventoryModels = []
            stableAudioCachedFinetunes = []
            stableAudioPredownloadCheckpointDownloaded = [:]
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
            modelDownloadStatusMessage = "downloading required carey files..."
        } else if canRunCareyFocusedDownload {
            let familyDescription = careyUseXlModels ? "xl" : "regular"
            modelDownloadStatusMessage = "download the \(familyDescription) carey model family you want to use, or fetch all three at once."
        } else {
            modelDownloadStatusMessage = "focused download script not found (expected in runtime/scripts or workspace/scripts)."
        }
    }

    func refreshCareyPredownloadInventory() {
        guard modelDownloadServiceID == "carey" else { return }
        guard let runtime = manager?.services.first(where: { $0.id == "carey" }) else {
            careyRequiredModels = []
            return
        }

        let fileManager = FileManager.default
        let checkpointDirectory = resolveCareyCheckpointDirectory(for: runtime)
        let requiredModelFiles = activeCareyRequiredModelFiles()
        let rows = requiredModelFiles.map { row in
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

        careyRequiredModels = rows
        if !isCareyDownloadInProgress {
            let downloadedCount = rows.filter(\.downloaded).count
            if downloadedCount == rows.count {
                modelDownloadStatusMessage = "all required carey files are downloaded."
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
            modelDownloadStatusMessage = "start \(serviceDisplayName) to download models."
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
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.stableAudioTokenInput = ""
                    self.applyStableAudioTokenState(configured: true)
                    self.stableAudioTokenStatus = "token saved in keychain."
                }
            } catch {
                DispatchQueue.main.async {
                    self?.stableAudioTokenStatus = error.localizedDescription
                }
            }
        }
    }

    func clearStableAudioToken() {
        stableAudioTokenStatus = "removing token..."
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                try StableAudioAuthKeychain.deleteToken()
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.stableAudioTokenInput = ""
                    self.applyStableAudioTokenState(configured: false)
                    self.stableAudioTokenStatus = "saved token removed."
                }
            } catch {
                DispatchQueue.main.async {
                    self?.stableAudioTokenStatus = error.localizedDescription
                }
            }
        }
    }

    func refreshStableAudioTokenState() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let configured = StableAudioAuthKeychain.readToken()?.isEmpty == false
            DispatchQueue.main.async {
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
                    }
                    self.modelDownloadStatusMessage = statusMessage

                    if pollResponse.status == "completed" {
                        self.isModelDownloadInProgress = false
                        self.activeModelDownloadPath = nil
                        self.activeModelDownloadSessionID = nil
                        self.modelDownloadPollTask = nil
                        if self.modelDownloadServiceID == serviceID,
                           serviceID != "stable_audio" {
                            self.refreshModelCatalogAndStatuses()
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
                self?.careyDownloadTask?.cancel()
                self?.manager?.shutdownForApplicationTermination()
            }
            .store(in: &cancellables)
    }
}
