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

@MainActor
final class ControlCenterViewModel: ObservableObject {
    private static let stableAudioBackendDefaultsKey = "stableAudioBackendEngine"
    private static let melodyFlowBackendDefaultsKey = "melodyFlowBackendEngine"

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
    @Published var stableAudioBackendEngine: StableAudioBackendEngine = .mps
    @Published var melodyFlowBackendEngine: MelodyFlowBackendEngine = .mps
    @Published var melodyFlowBackendStatus: String = ""
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
        observeApplicationTermination()
        refreshStableAudioTokenState()
        loadManifest()
    }

    deinit {
        logRefreshTask?.cancel()
        modelDownloadPollTask?.cancel()
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

    func loadManifest() {
        modelDownloadPollTask?.cancel()
        modelDownloadPollTask = nil
        isModelDownloadInProgress = false
        activeModelDownloadPath = nil
        activeModelDownloadSessionID = nil
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
        if stageTotal == 5 {
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
                self?.manager?.shutdownForApplicationTermination()
            }
            .store(in: &cancellables)
    }
}
