import Foundation
import Combine

struct SA3LoraTrainingRequest {
    let name: String
    let datasetPath: String
    let triggerText: String
    let steps: Int
    let rank: Int
    let adapterType: String
    let ditEngine: String
    let layerScope: String
    let fullTracks: Bool
    let cropSeconds: Double
    let learningRate: Double
    let saveEvery: Int
    let loudnessFixEnabled: Bool
    let targetLatentRMS: Double
}

struct SA3LoraTrainingState: Decodable, Equatable {
    let jobId: String?
    let name: String?
    let status: String?
    let phase: String?
    let message: String?
    let error: String?
    let runDir: String?
    let logPath: String?
    let cancelPath: String?
    let finalCheckpointPath: String?
    let currentStep: Int?
    let maxSteps: Int?
    let adapterType: String?
    let ditEngine: String?
    let layerScope: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case name
        case status
        case phase
        case message
        case error
        case runDir = "run_dir"
        case logPath = "log_path"
        case cancelPath = "cancel_path"
        case finalCheckpointPath = "final_checkpoint_path"
        case currentStep = "current_step"
        case maxSteps = "max_steps"
        case adapterType = "adapter_type"
        case ditEngine = "dit_engine"
        case layerScope = "layer_scope"
    }

    var isActive: Bool {
        guard let status else { return false }
        return ["starting", "running", "cancelling"].contains(status)
    }
}

@MainActor
final class SA3LoraTrainingManager: ObservableObject {
    @Published private(set) var state: SA3LoraTrainingState?
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
        request: SA3LoraTrainingRequest,
        service: ResolvedService,
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
            .appendingPathComponent("scripts/train_mlx_lora_job.py")

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
        guard fileManager.isExecutableFile(atPath: service.executable.path) else {
            launchError = "sa3 python is missing. rebuild the sa3 environment first."
            return
        }
        guard fileManager.isReadableFile(atPath: trainerURL.path) else {
            launchError = "mlx training runner is missing from the sa3 runtime."
            return
        }
        guard request.steps > 0, request.rank > 0, request.cropSeconds > 0,
              request.learningRate > 0, request.saveEvery >= 0,
              !request.loudnessFixEnabled
                || (0.5...1.3).contains(request.targetLatentRMS) else {
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
        let loraDirectory = URL(
            fileURLWithPath: service.environment["SA3_LORA_DIR"]
                ?? storageRoot.appendingPathComponent("loras").path
        )
        let registryURL = URL(
            fileURLWithPath: service.environment["SA3_LORA_REGISTRY"]
                ?? storageRoot.appendingPathComponent("lora_registry.json").path
        )
        let promptsDirectory = URL(
            fileURLWithPath: service.environment["SA3_PROMPTS_DIR"]
                ?? storageRoot.appendingPathComponent("prompts").path
        )
        let catalogURL = registryURL.deletingLastPathComponent()
            .appendingPathComponent("lora_catalog.json")

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
            process.executableURL = service.executable
            process.currentDirectoryURL = service.workingDirectory
            var arguments = [
                trainerURL.path,
                "--job-id", jobID,
                "--name", cleanName,
                "--dataset-dir", datasetURL.path,
                "--trigger-text", request.triggerText,
                "--steps", String(request.steps),
                "--rank", String(request.rank),
                "--adapter-type", request.adapterType,
                "--dit-engine", request.ditEngine,
                "--layer-scope", request.layerScope,
                "--crop-seconds", String(request.cropSeconds),
                "--learning-rate", String(request.learningRate),
                "--save-every", String(request.saveEvery),
                "--output-dir", runURL.path,
                "--status-path", statusURL.path,
                "--log-path", logURL.path,
                "--cancel-path", cancelURL.path,
                "--lora-dir", loraDirectory.path,
                "--registry-path", registryURL.path,
                "--catalog-path", catalogURL.path,
                "--prompts-dir", promptsDirectory.path,
            ]
            if request.fullTracks {
                arguments.append("--full-tracks")
            }
            if request.loudnessFixEnabled {
                arguments.append(contentsOf: [
                    "--per-track-target-latent-rms",
                    String(request.targetLatentRMS),
                ])
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
            state = SA3LoraTrainingState(
                jobId: jobID,
                name: cleanName,
                status: "starting",
                phase: "preparing",
                message: "Launching the MLX training process.",
                error: nil,
                runDir: runURL.path,
                logPath: logURL.path,
                cancelPath: cancelURL.path,
                finalCheckpointPath: nil,
                currentStep: 0,
                maxSteps: request.steps,
                adapterType: request.adapterType,
                ditEngine: request.ditEngine,
                layerScope: request.layerScope
            )
            logText = "Launching the MLX training process...\n"

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
        if state?.isActive == true || process?.isRunning == true {
            startPolling()
        }
    }

    private func refreshFromDisk() {
        if let statusURL,
           let data = try? Data(contentsOf: statusURL),
           let decoded = try? JSONDecoder().decode(SA3LoraTrainingState.self, from: data) {
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
            let latestLog = Self.readTextTail(from: logURL, maxBytes: 64 * 1024)
            if logText != latestLog {
                logText = latestLog
            }
        }
    }

    func clearError() {
        launchError = nil
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

    private static var storageRoot: URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Application Support/GaryLocalhost/sa3")
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
        return String((trimmed.isEmpty ? "sa3-lora" : trimmed).prefix(64))
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
