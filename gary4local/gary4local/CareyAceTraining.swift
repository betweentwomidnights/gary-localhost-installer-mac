import Foundation
import Combine
import Darwin

struct CareyAceTrainingRequest {
    let name: String
    let datasetPath: String
    let trigger: String
    let model: String
    let caption: String
    let captionLmModel: String
    let captionLmBackend: String
    let overwriteCaptions: Bool
    let bpmAnalysis: Bool
    let keyAnalysis: Bool
    let analysisDuration: Double
    let rank: Int
    let alpha: Int
    let adapterType: String
    let moduleProfile: String
    let timestepMu: Double
    let epochs: Int
    let maxSteps: Int
    let saveEvery: Int
    let saveBestAfter: Int
    let batchSize: Int
    let gradientAccumulation: Int
    let learningRate: Double
    let weightDecay: Double
    let cfgRatio: Double
    let lossWeighting: String
    let snrGamma: Double
    let maxDuration: Double
    let genreRatio: Int
    let preprocessDevice: String
    let preprocessPrecision: String
    let dtype: String
    let prepareOnly: Bool
}

struct CareyAceTrainingState: Decodable, Equatable {
    let jobId: String?
    let name: String?
    let status: String?
    let phase: String?
    let message: String?
    let error: String?
    let pid: Int?
    let childPid: Int?
    let runDir: String?
    let logPath: String?
    let cancelPath: String?
    let finalCheckpointPath: String?
    let bestCheckpointPath: String?
    let lastEpochCheckpointPath: String?
    let currentStep: Int?
    let maxSteps: Int?
    let currentEpoch: Int?
    let maxEpochs: Int?
    let currentLoss: Double?
    let currentFile: Int?
    let totalFiles: Int?
    let captionedCount: Int?
    let captionLmModel: String?
    let captionLmBackend: String?
    let adapterType: String?
    let moduleProfile: String?
    let modelFamily: String?
    let sampleCount: Int?
    let captionsPath: String?
    let datasetJsonPath: String?
    let trainingPlanPath: String?
    let resultPath: String?
    let registeredLoraName: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case name
        case status
        case phase
        case message
        case error
        case pid
        case childPid = "child_pid"
        case runDir = "run_dir"
        case logPath = "log_path"
        case cancelPath = "cancel_path"
        case finalCheckpointPath = "final_checkpoint_path"
        case bestCheckpointPath = "best_checkpoint_path"
        case lastEpochCheckpointPath = "last_epoch_checkpoint_path"
        case currentStep = "current_step"
        case maxSteps = "max_steps"
        case currentEpoch = "current_epoch"
        case maxEpochs = "max_epochs"
        case currentLoss = "current_loss"
        case currentFile = "current_file"
        case totalFiles = "total_files"
        case captionedCount = "captioned_count"
        case captionLmModel = "caption_lm_model"
        case captionLmBackend = "caption_lm_backend"
        case adapterType = "adapter_type"
        case moduleProfile = "module_profile"
        case modelFamily = "model_family"
        case sampleCount = "sample_count"
        case captionsPath = "captions_path"
        case datasetJsonPath = "dataset_json_path"
        case trainingPlanPath = "training_plan_path"
        case resultPath = "result_path"
        case registeredLoraName = "registered_lora_name"
    }

    var isActive: Bool {
        guard let status else { return false }
        return ["starting", "running", "cancelling"].contains(status)
    }
}

@MainActor
final class CareyAceTrainingManager: ObservableObject {
    @Published private(set) var state: CareyAceTrainingState?
    @Published private(set) var logText = ""
    @Published private(set) var launchError: String?
    @Published private(set) var isLaunching = false

    private struct CurrentJobPointer: Codable {
        let jobId: String
        let statusPath: String
        let logPath: String
        let cancelPath: String
        let runDir: String
    }

    private var process: Process?
    private var refreshTask: Task<Void, Never>?
    private var statusURL: URL?
    private var logURL: URL?
    private var cancelURL: URL?

    init() {
        loadCurrentJob()
    }

    deinit {
        refreshTask?.cancel()
    }

    func start(
        request: CareyAceTrainingRequest,
        service: ResolvedService,
        pythonURL: URL,
        checkpointDirectory: URL,
        loraCatalogURL: URL,
        loraRegistryURL: URL,
        captionsURL: URL,
        huggingFaceToken: String?
    ) {
        guard !isLaunching, process?.isRunning != true, state?.isActive != true else {
            launchError = "a training job is already running."
            return
        }

        let fileManager = FileManager.default
        let cleanName = Self.slugify(request.name)
        let datasetURL = URL(fileURLWithPath: request.datasetPath).standardizedFileURL
        let trainerURL = service.workingDirectory
            .appendingPathComponent("train_mlx_lora_job.py")

        guard !request.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            launchError = "enter a name for the lora."
            return
        }
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: datasetURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            launchError = "choose a readable audio folder."
            return
        }
        guard fileManager.fileExists(atPath: pythonURL.path) else {
            launchError = "carey python is missing. rebuild the carey environment first."
            return
        }
        guard fileManager.isReadableFile(atPath: trainerURL.path) else {
            launchError = "ace mlx training runner is missing from the carey runtime."
            return
        }
        guard fileManager.fileExists(atPath: checkpointDirectory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            launchError = "carey checkpoints are missing. download the carey models first."
            return
        }
        guard request.rank > 0, request.alpha > 0, request.epochs > 0,
              request.maxSteps >= 0,
              request.saveEvery > 0, request.saveBestAfter > 0,
              request.batchSize > 0, request.gradientAccumulation > 0,
              request.learningRate > 0, request.weightDecay >= 0,
              request.timestepMu.isFinite,
              (0..<1).contains(request.cfgRatio), request.snrGamma > 0,
              request.maxDuration > 0, (0...100).contains(request.genreRatio) else {
            launchError = "training settings must be positive numbers."
            return
        }

        let storageRoot = Self.storageRoot
        let trainingRoot = storageRoot.appendingPathComponent("training")
        let jobID = "\(Self.jobTimestamp())-\(cleanName)"
        let runURL = trainingRoot.appendingPathComponent("jobs/\(jobID)")
        let statusURL = runURL.appendingPathComponent("status.json")
        let logURL = runURL.appendingPathComponent("training.log")
        let cancelURL = runURL.appendingPathComponent("cancel.requested")
        let currentJobURL = trainingRoot.appendingPathComponent("current-job.json")

        do {
            try fileManager.createDirectory(at: runURL, withIntermediateDirectories: true)
            let pointer = CurrentJobPointer(
                jobId: jobID,
                statusPath: statusURL.path,
                logPath: logURL.path,
                cancelPath: cancelURL.path,
                runDir: runURL.path
            )
            try JSONEncoder().encode(pointer).write(to: currentJobURL, options: .atomic)

            let process = Process()
            process.executableURL = pythonURL
            process.currentDirectoryURL = service.workingDirectory
            var arguments = [
                trainerURL.path,
                "--job-id", jobID,
                "--name", cleanName,
                "--dataset-dir", datasetURL.path,
                "--checkpoint-dir", checkpointDirectory.path,
                "--run-dir", runURL.path,
                "--status-path", statusURL.path,
                "--current-job-path", currentJobURL.path,
                "--log-path", logURL.path,
                "--cancel-path", cancelURL.path,
                "--model", request.model,
                "--trigger", request.trigger,
                "--tag-position", "prepend",
                "--genre-ratio", String(request.genreRatio),
                "--caption", request.caption,
                "--carey-url", "http://127.0.0.1:8013",
                "--inference-carey-url", "http://127.0.0.1:8003",
                "--caption-lm-model", request.captionLmModel,
                "--caption-lm-backend", request.captionLmBackend,
                "--analysis-duration", String(request.analysisDuration),
                "--rank", String(request.rank),
                "--alpha", String(request.alpha),
                "--adapter-type", request.adapterType,
                "--module-profile", request.moduleProfile,
                "--learning-rate", String(request.learningRate),
                "--timestep-mu", String(request.timestepMu),
                "--weight-decay", String(request.weightDecay),
                "--cfg-ratio", String(request.cfgRatio),
                "--loss-weighting", request.lossWeighting,
                "--snr-gamma", String(request.snrGamma),
                "--epochs", String(request.epochs),
                "--save-every", String(request.saveEvery),
                "--save-best-after", String(request.saveBestAfter),
                "--batch-size", String(request.batchSize),
                "--gradient-accumulation", String(request.gradientAccumulation),
                "--max-duration", String(request.maxDuration),
                "--preprocess-device", request.preprocessDevice,
                "--preprocess-precision", request.preprocessPrecision,
                "--dtype", request.dtype,
                "--lora-catalog-path", loraCatalogURL.path,
                "--lora-registry-path", loraRegistryURL.path,
                "--captions-json-path", captionsURL.path,
            ]
            if request.overwriteCaptions {
                arguments.append("--overwrite-captions")
            }
            if !request.bpmAnalysis {
                arguments.append("--no-bpm-analysis")
            }
            if !request.keyAnalysis {
                arguments.append("--no-key-analysis")
            }
            if request.maxSteps > 0 {
                arguments.append(contentsOf: ["--max-steps", String(request.maxSteps)])
            }
            if request.model == "xl-base" {
                arguments.append("--allow-unsafe-xl")
            }
            if request.prepareOnly {
                arguments.append("--prepare-only")
            }
            process.arguments = arguments
            process.environment = Self.environment(
                for: service,
                huggingFaceToken: huggingFaceToken
            )
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice

            self.statusURL = statusURL
            self.logURL = logURL
            self.cancelURL = cancelURL
            self.process = process
            launchError = nil
            isLaunching = true
            state = CareyAceTrainingState(
                jobId: jobID,
                name: cleanName,
                status: "starting",
                phase: "preparing",
                message: "Launching the ACE MLX training process.",
                error: nil,
                pid: nil,
                childPid: nil,
                runDir: runURL.path,
                logPath: logURL.path,
                cancelPath: cancelURL.path,
                finalCheckpointPath: nil,
                bestCheckpointPath: nil,
                lastEpochCheckpointPath: nil,
                currentStep: 0,
                maxSteps: nil,
                currentEpoch: 0,
                maxEpochs: request.epochs,
                currentLoss: nil,
                currentFile: nil,
                totalFiles: nil,
                captionedCount: nil,
                captionLmModel: request.caption == "understand_music" ? request.captionLmModel : nil,
                captionLmBackend: request.caption == "understand_music" ? request.captionLmBackend : nil,
                adapterType: request.adapterType,
                moduleProfile: request.moduleProfile,
                modelFamily: request.model == "xl-base" ? "xl" : "standard",
                sampleCount: nil,
                captionsPath: nil,
                datasetJsonPath: nil,
                trainingPlanPath: nil,
                resultPath: nil,
                registeredLoraName: nil
            )
            logText = "Launching the ACE MLX training process...\n"

            try process.run()
            isLaunching = false
            startPolling()
        } catch {
            isLaunching = false
            process = nil
            state = nil
            launchError = error.localizedDescription
        }
    }

    func cancel() {
        guard let cancelURL, state?.isActive == true else { return }
        do {
            try Data("\(ISO8601DateFormatter().string(from: Date()))\n".utf8)
                .write(to: cancelURL, options: .atomic)
            launchError = nil
            refresh()
        } catch {
            launchError = "could not request cancellation: \(error.localizedDescription)"
        }
    }

    func refresh() {
        refreshFromDisk()
        clearStaleActiveStateIfNeeded()
        if state?.isActive == true || process?.isRunning == true {
            startPolling()
        }
    }

    func clearError() {
        launchError = nil
    }

    func reportLaunchError(_ message: String) {
        launchError = message
    }

    private func refreshFromDisk() {
        if let statusURL,
           let data = try? Data(contentsOf: statusURL),
           let decoded = try? JSONDecoder().decode(CareyAceTrainingState.self, from: data) {
            if state != decoded {
                state = decoded
            }
            if let path = decoded.logPath {
                logURL = URL(fileURLWithPath: path)
            }
            if let path = decoded.cancelPath {
                cancelURL = URL(fileURLWithPath: path)
            }
        }
        if let logURL {
            let latestLog = Self.readTextTail(from: logURL, maxBytes: 96 * 1024)
            if logText != latestLog {
                logText = latestLog
            }
        }
    }

    private func loadCurrentJob() {
        let pointerURL = Self.storageRoot
            .appendingPathComponent("training/current-job.json")
        guard let data = try? Data(contentsOf: pointerURL),
              let pointer = try? JSONDecoder().decode(CurrentJobPointer.self, from: data) else {
            return
        }
        statusURL = URL(fileURLWithPath: pointer.statusPath)
        logURL = URL(fileURLWithPath: pointer.logPath)
        cancelURL = URL(fileURLWithPath: pointer.cancelPath)
        refresh()
    }

    private func startPolling() {
        guard refreshTask == nil else { return }
        refreshTask = Task { [weak self] in
            var exitedWhileActivePasses = 0
            while !Task.isCancelled {
                guard let self else { return }
                self.refreshFromDisk()
                self.clearStaleActiveStateIfNeeded()
                let hasManagedProcess = self.process != nil
                let processIsRunning = self.process?.isRunning == true

                if hasManagedProcess, !processIsRunning, self.state?.isActive == true {
                    exitedWhileActivePasses += 1
                    if exitedWhileActivePasses >= 4 {
                        self.launchError = "the training runner exited before reporting completion."
                        self.state = nil
                        break
                    }
                } else {
                    exitedWhileActivePasses = 0
                }

                if !hasManagedProcess, self.state?.isActive != true {
                    break
                }
                if hasManagedProcess, !processIsRunning, self.state?.isActive != true {
                    break
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            if self?.process?.isRunning == false {
                self?.process = nil
            }
            self?.refreshTask = nil
        }
    }

    private func clearStaleActiveStateIfNeeded() {
        guard process == nil, let state, state.isActive else { return }
        if Self.processExists(state.pid) || Self.processExists(state.childPid) {
            return
        }
        launchError = "the previous ACE training job was interrupted before it could report completion."
        self.state = nil
    }

    private static func processExists(_ pid: Int?) -> Bool {
        guard let pid, pid > 0 else { return false }
        if kill(pid_t(pid), 0) == 0 {
            return true
        }
        return errno == EPERM
    }

    private static var storageRoot: URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Application Support/GaryLocalhost/carey")
    }

    private static func environment(
        for service: ResolvedService,
        huggingFaceToken: String?
    ) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        for key in [
            "DYLD_INSERT_LIBRARIES",
            "METAL_DEVICE_WRAPPER_TYPE",
            "METAL_DEBUG_ERROR_MODE",
            "__XPC_DYLD_FRAMEWORK_PATH",
            "__XPC_DYLD_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
            "DYLD_LIBRARY_PATH",
            "__XCODE_BUILT_PRODUCTS_DIR_PATHS",
            "SWIFTUI_VIEW_DEBUG",
        ] {
            environment.removeValue(forKey: key)
        }
        environment["PYTHONUNBUFFERED"] = "1"
        for (key, value) in service.environment {
            environment[key] = value
        }
        if let token = huggingFaceToken?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty {
            environment["HF_TOKEN"] = token
            environment["HUGGING_FACE_HUB_TOKEN"] = token
        }
        return environment
    }

    private static func slugify(_ raw: String) -> String {
        let lower = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-"))
        var output = ""
        var previousWasDash = false
        for scalar in lower.unicodeScalars {
            if allowed.contains(scalar) {
                output.unicodeScalars.append(scalar)
                previousWasDash = false
            } else if !previousWasDash {
                output.append("-")
                previousWasDash = true
            }
        }
        let trimmed = output.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return String((trimmed.isEmpty ? "ace-lora" : trimmed).prefix(64))
    }

    private static func jobTimestamp() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    private static func readTextTail(from url: URL, maxBytes: Int) -> String {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = (attributes[.size] as? NSNumber)?.intValue,
              fileSize > 0,
              let handle = try? FileHandle(forReadingFrom: url) else {
            return ""
        }
        defer { try? handle.close() }
        let readSize = min(fileSize, maxBytes)
        if fileSize > readSize {
            try? handle.seek(toOffset: UInt64(fileSize - readSize))
        }
        guard let data = try? handle.read(upToCount: readSize) else { return "" }
        var text = String(decoding: data, as: UTF8.self)
        if fileSize > readSize, let newline = text.firstIndex(of: "\n") {
            text = String(text[text.index(after: newline)...])
        }
        return text
    }
}
