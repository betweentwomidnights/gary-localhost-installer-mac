import Foundation
import Combine
import Darwin

enum SA3PromptStyle: String, CaseIterable, Identifiable {
    case bare
    case labeled

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .bare: return "barebones"
        case .labeled: return "official SA3"
        }
    }
}

struct SA3MetadataSuggestion: Decodable {
    let ok: Bool
    let bpm: Int?
    let keyscale: String?
    let bpmConfidence: Double?
    let keyConfidence: Double?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case bpm
        case keyscale
        case bpmConfidence = "bpm_confidence"
        case keyConfidence = "key_confidence"
        case error
    }
}

enum SA3AudioMetadataAnalyzer {
    static func analyze(
        audioPath: String,
        service: ResolvedService,
        pythonURL: URL
    ) async throws -> SA3MetadataSuggestion {
        let scriptURL = service.workingDirectory
            .appendingPathComponent("scripts/analyze_audio.py")
        let environment = processEnvironment(for: service)
        let execution = try await Task.detached(priority: .userInitiated) {
            let fileManager = FileManager.default
            let audioURL = URL(fileURLWithPath: audioPath).standardizedFileURL
            guard fileManager.isReadableFile(atPath: audioURL.path) else {
                throw NSError(
                    domain: "SA3AudioMetadataAnalyzer",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "the selected audio file is not readable."
                    ]
                )
            }
            guard fileManager.fileExists(atPath: pythonURL.path) else {
                throw NSError(
                    domain: "SA3AudioMetadataAnalyzer",
                    code: 2,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "the SA3 Python environment is missing. Rebuild SA3 first."
                    ]
                )
            }
            guard fileManager.isReadableFile(atPath: scriptURL.path) else {
                throw NSError(
                    domain: "SA3AudioMetadataAnalyzer",
                    code: 3,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "the SA3 BPM/key analysis helper is missing."
                    ]
                )
            }

            let temporaryURL = fileManager.temporaryDirectory
                .appendingPathComponent("gary-sa3-analysis-\(UUID().uuidString)")
            try fileManager.createDirectory(
                at: temporaryURL,
                withIntermediateDirectories: true
            )
            defer { try? fileManager.removeItem(at: temporaryURL) }
            let stdoutURL = temporaryURL.appendingPathComponent("stdout.json")
            let stderrURL = temporaryURL.appendingPathComponent("stderr.log")
            fileManager.createFile(atPath: stdoutURL.path, contents: nil)
            fileManager.createFile(atPath: stderrURL.path, contents: nil)
            let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
            let stderrHandle = try FileHandle(forWritingTo: stderrURL)
            defer {
                try? stdoutHandle.close()
                try? stderrHandle.close()
            }

            let process = Process()
            process.executableURL = pythonURL
            process.currentDirectoryURL = service.workingDirectory
            process.arguments = [scriptURL.path, audioURL.path]
            process.environment = environment
            process.standardOutput = stdoutHandle
            process.standardError = stderrHandle
            try process.run()
            process.waitUntilExit()
            try? stdoutHandle.synchronize()
            try? stderrHandle.synchronize()

            let stdout = (try? Data(contentsOf: stdoutURL)) ?? Data()
            let stderr = String(
                decoding: (try? Data(contentsOf: stderrURL)) ?? Data(),
                as: UTF8.self
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            return (process.terminationStatus, stdout, stderr)
        }.value
        let result: SA3MetadataSuggestion
        do {
            result = try JSONDecoder().decode(
                SA3MetadataSuggestion.self,
                from: execution.1
            )
        } catch {
            throw failure(
                execution.2.isEmpty
                    ? "the analyzer did not return a valid result."
                    : execution.2
            )
        }
        guard execution.0 == 0, result.ok else {
            throw failure(
                result.error
                    ?? (execution.2.isEmpty ? "BPM/key analysis failed." : execution.2)
            )
        }
        return result
    }

    private static func processEnvironment(
        for service: ResolvedService
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
        return environment
    }

    private nonisolated static func failure(_ message: String) -> NSError {
        NSError(
            domain: "SA3AudioMetadataAnalyzer",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}

struct SA3AutolabelState: Decodable, Equatable {
    let jobId: String?
    let status: String?
    let phase: String?
    let message: String?
    let error: String?
    let pid: Int?
    let childPid: Int?
    let runDir: String?
    let logPath: String?
    let cancelPath: String?
    let datasetPath: String?
    let currentPath: String?
    let total: Int?
    let done: Int?
    let style: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
        case phase
        case message
        case error
        case pid
        case childPid = "child_pid"
        case runDir = "run_dir"
        case logPath = "log_path"
        case cancelPath = "cancel_path"
        case datasetPath = "dataset_path"
        case currentPath = "current_path"
        case total
        case done
        case style
    }

    var isActive: Bool {
        guard let status else { return false }
        return ["starting", "running", "cancelling"].contains(status)
    }

    var isTerminal: Bool {
        guard let status else { return false }
        return ["completed", "cancelled", "failed"].contains(status)
    }
}

@MainActor
final class SA3AutolabelManager: ObservableObject {
    @Published private(set) var state: SA3AutolabelState?
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
    private var logHandle: FileHandle?
    private var refreshTask: Task<Void, Never>?
    private var statusURL: URL?
    private var logURL: URL?
    private var cancelURL: URL?

    init() {
        loadCurrentJob()
    }

    deinit {
        refreshTask?.cancel()
        try? logHandle?.close()
    }

    func start(
        datasetPath: String,
        style: SA3PromptStyle,
        service: ResolvedService,
        pythonURL: URL,
        huggingFaceToken: String?,
        careyServiceIsRunning: Bool,
        careyTrainingIsActive: Bool
    ) {
        guard !isLaunching, process?.isRunning != true, state?.isActive != true else {
            launchError = "an SA3 auto-label job is already running."
            return
        }
        guard !careyServiceIsRunning else {
            launchError = "stop Carey before auto-labeling this dataset."
            return
        }
        guard !careyTrainingIsActive else {
            launchError = "wait for the active Carey training job to finish."
            return
        }

        let fileManager = FileManager.default
        let datasetURL = URL(fileURLWithPath: datasetPath).standardizedFileURL
        let scriptURL = service.workingDirectory
            .appendingPathComponent("sa3_autolabel.py")
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: datasetURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            launchError = "choose a valid audio folder first."
            return
        }
        guard fileManager.fileExists(atPath: pythonURL.path) else {
            launchError = "the Carey Python environment is missing. Rebuild Carey first."
            return
        }
        guard fileManager.isReadableFile(atPath: scriptURL.path) else {
            launchError = "the Carey SA3 auto-label helper is missing."
            return
        }

        let root = Self.storageRoot
        let jobID = "\(Self.jobTimestamp())-\(UUID().uuidString.prefix(8))"
        let runURL = root.appendingPathComponent("jobs/\(jobID)")
        let statusURL = runURL.appendingPathComponent("status.json")
        let logURL = runURL.appendingPathComponent("autolabel.log")
        let cancelURL = runURL.appendingPathComponent("cancel.requested")
        let currentJobURL = root.appendingPathComponent("current-job.json")

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
            fileManager.createFile(atPath: logURL.path, contents: nil)
            let logHandle = try FileHandle(forWritingTo: logURL)

            let process = Process()
            process.executableURL = pythonURL
            process.currentDirectoryURL = service.workingDirectory
            process.arguments = [
                scriptURL.path,
                "--dataset-dir", datasetURL.path,
                "--style", style.rawValue,
                "--caption-lm-model", "acestep-5Hz-lm-1.7B",
                "--caption-lm-backend", "mlx",
                "--job-id", jobID,
                "--run-dir", runURL.path,
                "--status-path", statusURL.path,
                "--current-job-path", currentJobURL.path,
                "--log-path", logURL.path,
                "--cancel-path", cancelURL.path,
            ]
            process.environment = Self.environment(
                for: service,
                huggingFaceToken: huggingFaceToken
            )
            process.standardOutput = logHandle
            process.standardError = logHandle

            self.statusURL = statusURL
            self.logURL = logURL
            self.cancelURL = cancelURL
            self.process = process
            self.logHandle = logHandle
            launchError = nil
            isLaunching = true
            state = SA3AutolabelState(
                jobId: jobID,
                status: "starting",
                phase: "starting",
                message: "Launching the Carey captioner.",
                error: nil,
                pid: nil,
                childPid: nil,
                runDir: runURL.path,
                logPath: logURL.path,
                cancelPath: cancelURL.path,
                datasetPath: datasetURL.path,
                currentPath: nil,
                total: nil,
                done: 0,
                style: style.rawValue
            )
            logText = "Launching the Carey captioner...\n"
            try process.run()
            isLaunching = false
            startPolling()
        } catch {
            isLaunching = false
            process = nil
            try? logHandle?.close()
            logHandle = nil
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

    func reportLaunchError(_ message: String) {
        launchError = message
    }

    func clearError() {
        launchError = nil
    }

    private func refreshFromDisk() {
        if let statusURL,
           let data = try? Data(contentsOf: statusURL),
           let decoded = try? JSONDecoder().decode(SA3AutolabelState.self, from: data) {
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
            let latest = Self.readTextTail(from: logURL, maxBytes: 96 * 1024)
            if logText != latest {
                logText = latest
            }
        }
    }

    private func loadCurrentJob() {
        let pointerURL = Self.storageRoot.appendingPathComponent("current-job.json")
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
                        self.launchError =
                            "the auto-label runner exited before reporting completion."
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
                try? self?.logHandle?.close()
                self?.logHandle = nil
            }
            self?.refreshTask = nil
        }
    }

    private func clearStaleActiveStateIfNeeded() {
        guard process == nil, let state, state.isActive else { return }
        if Self.processExists(state.pid) || Self.processExists(state.childPid) {
            return
        }
        launchError = "the previous SA3 auto-label job was interrupted."
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
            .appendingPathComponent(
                "Library/Application Support/GaryLocalhost/sa3/autolabel",
                isDirectory: true
            )
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
