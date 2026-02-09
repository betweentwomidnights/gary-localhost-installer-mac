import Foundation
import Darwin
import Combine

private struct BootstrapRunResult {
    let success: Bool
    let message: String
}

private enum BootstrapError: LocalizedError {
    case pythonNotFound(executable: String, searchedPaths: [String])

    var errorDescription: String? {
        switch self {
        case .pythonNotFound(let executable, let searchedPaths):
            let paths = searchedPaths.joined(separator: ", ")
            return "Python executable '\(executable)' was not found. Searched: \(paths)"
        }
    }
}

private enum ServiceStartError: LocalizedError {
    case portStillInUse(port: Int, pids: [Int32])
    case missingStableAudioToken

    var errorDescription: String? {
        switch self {
        case .portStillInUse(let port, let pids):
            let pidList = pids.map(String.init).joined(separator: ", ")
            return "Port \(port) is still in use by PID(s): \(pidList)"
        case .missingStableAudioToken:
            return "Stable Audio requires Hugging Face access token. Open Stable Audio setup and save your token."
        }
    }
}

@MainActor
final class ServiceManager: ObservableObject {
    @Published private(set) var services: [ServiceRuntime]
    @Published private(set) var isRebuildingAllEnvironments = false

    private let manifest: ResolvedManifest
    private var processes: [String: Process] = [:]
    private var outputPipes: [String: Pipe] = [:]
    private var logHandles: [String: FileHandle] = [:]
    private var healthTasks: [String: Task<Void, Never>] = [:]
    private var bootstrapTasks: [String: Task<Void, Never>] = [:]
    private var rebuildAllTask: Task<Void, Never>?

    init(manifest: ResolvedManifest) {
        self.manifest = manifest
        self.services = manifest.services.map { ServiceRuntime(service: $0) }
    }

    deinit {
        rebuildAllTask?.cancel()
        for (_, task) in healthTasks {
            task.cancel()
        }
        for (_, task) in bootstrapTasks {
            task.cancel()
        }
        for (_, process) in processes where process.isRunning {
            process.terminate()
        }
        for (_, handle) in logHandles {
            try? handle.close()
        }
    }

    func startAutoStartServices() {
        for runtime in services where runtime.service.autoStart {
            start(serviceID: runtime.service.id)
        }
    }

    func start(serviceID: String) {
        guard let index = indexForService(serviceID) else { return }
        guard processes[serviceID]?.isRunning != true else { return }
        guard bootstrapTasks[serviceID] == nil else {
            services[index].lastError = "Environment rebuild in progress."
            return
        }

        let service = services[index].service
        services[index].processState = .starting
        services[index].lastError = nil
        services[index].healthState = .unknown

        do {
            let environment = makeEnvironment(for: service)
            if service.id == "stable_audio", environment["HF_TOKEN"]?.isEmpty != false {
                throw ServiceStartError.missingStableAudioToken
            }
            try clearConflictingListenersIfNeeded(
                for: service,
                environment: environment,
                failIfNotCleared: true
            )

            let logFileURL = try prepareLogFile(for: service)
            let logHandle = try FileHandle(forWritingTo: logFileURL)
            logHandle.seekToEndOfFile()

            let pipe = Pipe()
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty else { return }
                logHandle.write(data)
            }

            let process = Process()
            process.executableURL = service.executable
            process.currentDirectoryURL = service.workingDirectory
            process.arguments = service.arguments
            process.standardOutput = pipe
            process.standardError = pipe
            process.environment = environment
            process.terminationHandler = { [weak self] terminated in
                DispatchQueue.main.async {
                    self?.handleTermination(serviceID: serviceID, process: terminated)
                }
            }

            try process.run()

            outputPipes[serviceID] = pipe
            logHandles[serviceID] = logHandle
            processes[serviceID] = process

            services[index].processState = .running
            services[index].pid = process.processIdentifier
            services[index].lastExitCode = nil
            services[index].healthState = .unknown

            startHealthMonitor(for: service)
        } catch {
            services[index].processState = .failed
            services[index].lastError = error.localizedDescription
            cleanupIO(for: serviceID)
        }
    }

    func stop(serviceID: String) {
        guard let index = indexForService(serviceID) else { return }
        guard let process = processes[serviceID] else {
            let service = services[index].service
            let environment = makeEnvironment(for: service)
            do {
                try clearConflictingListenersIfNeeded(
                    for: service,
                    environment: environment,
                    failIfNotCleared: false
                )
            } catch {
                services[index].lastError = error.localizedDescription
            }
            services[index].processState = .stopped
            services[index].healthState = .unknown
            return
        }

        services[index].processState = .stopping
        healthTasks[serviceID]?.cancel()
        healthTasks[serviceID] = nil

        process.terminate()

        let grace = services[index].service.gracefulShutdownSeconds
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(grace) * 1_000_000_000)
            guard let self else { return }
            guard let stillRunning = self.processes[serviceID], stillRunning.isRunning else { return }
            kill(stillRunning.processIdentifier, SIGKILL)
        }
    }

    func restart(serviceID: String) {
        stop(serviceID: serviceID)
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            await MainActor.run {
                self?.start(serviceID: serviceID)
            }
        }
    }

    func refreshHealthNow(serviceID: String) {
        guard let service = services.first(where: { $0.id == serviceID })?.service else { return }
        startHealthMonitor(for: service)
    }

    func rebuildEnvironment(serviceID: String) {
        guard let index = indexForService(serviceID) else { return }
        guard bootstrapTasks[serviceID] == nil else { return }

        guard processes[serviceID]?.isRunning != true else {
            services[index].bootstrapState = .failed
            services[index].bootstrapMessage = "Stop the service before rebuilding its environment."
            return
        }

        guard let bootstrap = services[index].service.bootstrap else {
            services[index].bootstrapState = .notConfigured
            services[index].bootstrapMessage = "No bootstrap configuration found in manifest."
            return
        }

        services[index].bootstrapState = .running
        services[index].bootstrapMessage = "Rebuilding environment..."
        services[index].lastError = nil

        let service = services[index].service
        let inheritedEnvironment = makeEnvironment(for: service)

        let task = Task { [weak self] in
            let result = await Task.detached(priority: .userInitiated) {
                Self.runBootstrap(
                    service: service,
                    bootstrap: bootstrap,
                    inheritedEnvironment: inheritedEnvironment
                )
            }.value

            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self else { return }
                self.bootstrapTasks[serviceID] = nil
                guard let latestIndex = self.indexForService(serviceID) else { return }
                self.services[latestIndex].bootstrapState = result.success ? .succeeded : .failed
                self.services[latestIndex].bootstrapMessage = result.message
                if !result.success {
                    self.services[latestIndex].lastError = result.message
                }
            }
        }

        bootstrapTasks[serviceID] = task
    }

    func rebuildAllEnvironments() {
        guard rebuildAllTask == nil else { return }

        let serviceIDs = services.map(\.id)
        rebuildAllTask = Task { [weak self] in
            await MainActor.run {
                guard let self else { return }
                self.isRebuildingAllEnvironments = true
            }

            for serviceID in serviceIDs {
                guard !Task.isCancelled else { break }
                await MainActor.run {
                    self?.rebuildEnvironment(serviceID: serviceID)
                }

                while !Task.isCancelled {
                    let hasRunningTask = await MainActor.run { [weak self] in
                        guard let self else { return false }
                        return self.bootstrapTasks[serviceID] != nil
                    }
                    if !hasRunningTask {
                        break
                    }
                    try? await Task.sleep(nanoseconds: 250_000_000)
                }
            }

            await MainActor.run {
                guard let self else { return }
                self.isRebuildingAllEnvironments = false
                self.rebuildAllTask = nil
            }
        }
    }

    func readLogTail(
        serviceID: String,
        maxLines: Int = 220,
        maxBytes: Int = 256_000
    ) -> String {
        guard let service = services.first(where: { $0.id == serviceID })?.service else {
            return ""
        }
        guard maxLines > 0, maxBytes > 0 else {
            return ""
        }

        let fileManager = FileManager.default
        guard let attributes = try? fileManager.attributesOfItem(atPath: service.logFile.path),
              let fileSizeNumber = attributes[.size] as? NSNumber else {
            return ""
        }

        let fileSize = fileSizeNumber.intValue
        if fileSize == 0 {
            return ""
        }

        let readSize = min(maxBytes, fileSize)

        guard let handle = try? FileHandle(forReadingFrom: service.logFile) else {
            return ""
        }
        defer { try? handle.close() }

        if fileSize > readSize {
            try? handle.seek(toOffset: UInt64(fileSize - readSize))
        }

        guard let data = try? handle.read(upToCount: readSize),
              !data.isEmpty else {
            return ""
        }

        var text = String(decoding: data, as: UTF8.self)
        if fileSize > readSize, let firstNewline = text.firstIndex(of: "\n") {
            text = String(text[text.index(after: firstNewline)...])
        }

        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        let tail = lines.suffix(maxLines)
        return tail.joined(separator: "\n")
    }

    nonisolated private static func runBootstrap(
        service: ResolvedService,
        bootstrap: ResolvedBootstrapConfig,
        inheritedEnvironment: [String: String]
    ) -> BootstrapRunResult {
        do {
            let logURL = try prepareLogFile(at: service.logFile)
            let logHandle = try FileHandle(forWritingTo: logURL)
            defer { try? logHandle.close() }
            logHandle.seekToEndOfFile()

            writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild started ----", to: logHandle)

            let pythonExecutable = try resolvePythonExecutable(
                named: bootstrap.pythonExecutable,
                environment: inheritedEnvironment
            )
            writeLogLine("Using python executable: \(pythonExecutable.path)", to: logHandle)

            _ = try runCommand(
                executable: pythonExecutable,
                arguments: ["--version"],
                currentDirectory: service.workingDirectory,
                environment: inheritedEnvironment,
                logHandle: logHandle
            )

            let versionCheckStatus = try runCommand(
                executable: pythonExecutable,
                arguments: [
                    "-c",
                    "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
                ],
                currentDirectory: service.workingDirectory,
                environment: inheritedEnvironment,
                logHandle: logHandle
            )
            if versionCheckStatus != 0 {
                let message = "Python 3.11+ is required. Update bootstrap.python_executable for this service."
                writeLogLine(message, to: logHandle)
                writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild failed ----", to: logHandle)
                return BootstrapRunResult(success: false, message: message)
            }

            if !FileManager.default.fileExists(atPath: bootstrap.requirementsFile.path) {
                let message = "Requirements file not found: \(bootstrap.requirementsFile.path)"
                writeLogLine(message, to: logHandle)
                writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild failed ----", to: logHandle)
                return BootstrapRunResult(success: false, message: message)
            }

            let venvPython = bootstrap.venvDirectory.appendingPathComponent("bin/python")
            let fileManager = FileManager.default
            var shouldCreateVenv = !fileManager.fileExists(atPath: bootstrap.venvDirectory.path)

            if !shouldCreateVenv {
                writeLogLine("Using existing venv: \(bootstrap.venvDirectory.path)", to: logHandle)

                if !fileManager.fileExists(atPath: venvPython.path) {
                    writeLogLine("Existing venv is missing bin/python; recreating.", to: logHandle)
                    shouldCreateVenv = true
                } else {
                    let venvVersionStatus = try runCommand(
                        executable: venvPython,
                        arguments: [
                            "-c",
                            "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
                        ],
                        currentDirectory: service.workingDirectory,
                        environment: inheritedEnvironment,
                        logHandle: logHandle
                    )
                    if venvVersionStatus != 0 {
                        writeLogLine("Existing venv Python is below 3.11; recreating venv.", to: logHandle)
                        try? fileManager.removeItem(at: bootstrap.venvDirectory)
                        shouldCreateVenv = true
                    }
                }
            }

            if shouldCreateVenv {
                let createStatus = try runCommand(
                    executable: pythonExecutable,
                    arguments: ["-m", "venv", bootstrap.venvDirectory.path],
                    currentDirectory: service.workingDirectory,
                    environment: inheritedEnvironment,
                    logHandle: logHandle
                )
                if createStatus != 0 {
                    let message = "Failed to create venv (exit \(createStatus))."
                    writeLogLine(message, to: logHandle)
                    writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild failed ----", to: logHandle)
                    return BootstrapRunResult(success: false, message: message)
                }
            }

            guard fileManager.fileExists(atPath: venvPython.path) else {
                let message = "venv python missing at \(venvPython.path)"
                writeLogLine(message, to: logHandle)
                writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild failed ----", to: logHandle)
                return BootstrapRunResult(success: false, message: message)
            }

            if bootstrap.upgradeBuildTools {
                let upgradeStatus = try runCommand(
                    executable: venvPython,
                    arguments: ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                    currentDirectory: service.workingDirectory,
                    environment: inheritedEnvironment,
                    logHandle: logHandle
                )
                if upgradeStatus != 0 {
                    let message = "Failed to upgrade pip/setuptools/wheel (exit \(upgradeStatus))."
                    writeLogLine(message, to: logHandle)
                    writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild failed ----", to: logHandle)
                    return BootstrapRunResult(success: false, message: message)
                }
            }

            var installArguments = ["-m", "pip", "install"]
            installArguments.append(contentsOf: bootstrap.pipArguments)
            installArguments.append(contentsOf: ["-r", bootstrap.requirementsFile.path])

            let installStatus = try runCommand(
                executable: venvPython,
                arguments: installArguments,
                currentDirectory: service.workingDirectory,
                environment: inheritedEnvironment,
                logHandle: logHandle
            )
            if installStatus != 0 {
                let message = "Dependency install failed (exit \(installStatus))."
                writeLogLine(message, to: logHandle)
                writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild failed ----", to: logHandle)
                return BootstrapRunResult(success: false, message: message)
            }

            writeLogLine("---- \(timestamp()) [\(service.id)] Environment rebuild succeeded ----", to: logHandle)
            return BootstrapRunResult(
                success: true,
                message: "Environment rebuilt successfully at \(timestamp())."
            )
        } catch {
            return BootstrapRunResult(success: false, message: error.localizedDescription)
        }
    }

    nonisolated private static func resolvePythonExecutable(
        named executableName: String,
        environment: [String: String]
    ) throws -> URL {
        let expandedExecutable = NSString(string: executableName).expandingTildeInPath
        let fileManager = FileManager.default

        if expandedExecutable.contains("/") {
            let directURL = URL(fileURLWithPath: expandedExecutable).standardizedFileURL
            if fileManager.isExecutableFile(atPath: directURL.path) {
                return directURL
            }
            throw BootstrapError.pythonNotFound(
                executable: executableName,
                searchedPaths: [directURL.path]
            )
        }

        var searchPaths: [String] = []
        if let rawPath = environment["PATH"], !rawPath.isEmpty {
            searchPaths.append(contentsOf: rawPath.split(separator: ":").map(String.init))
        }

        searchPaths.append(contentsOf: [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin",
        ])

        var seen = Set<String>()
        let uniquePaths = searchPaths.filter { seen.insert($0).inserted }

        for path in uniquePaths {
            let candidateURL = URL(fileURLWithPath: path)
                .appendingPathComponent(expandedExecutable)
                .standardizedFileURL
            if fileManager.isExecutableFile(atPath: candidateURL.path) {
                return candidateURL
            }
        }

        throw BootstrapError.pythonNotFound(executable: executableName, searchedPaths: uniquePaths)
    }

    nonisolated private static func runCommand(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        environment: [String: String],
        logHandle: FileHandle
    ) throws -> Int32 {
        writeLogLine("$ \(executable.path) \(arguments.joined(separator: " "))", to: logHandle)

        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        process.environment = environment

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            logHandle.write(data)
        }

        try process.run()
        process.waitUntilExit()

        outputPipe.fileHandleForReading.readabilityHandler = nil
        writeLogLine("Exit code: \(process.terminationStatus)", to: logHandle)
        return process.terminationStatus
    }

    nonisolated private static func writeLogLine(_ line: String, to logHandle: FileHandle) {
        guard let data = "\(line)\n".data(using: .utf8) else { return }
        logHandle.write(data)
    }

    nonisolated private static func timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: Date())
    }

    nonisolated private static func prepareLogFile(at url: URL) throws -> URL {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        return url
    }

    private func prepareLogFile(for service: ResolvedService) throws -> URL {
        let directory = service.logFile.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: service.logFile.path) {
            FileManager.default.createFile(atPath: service.logFile.path, contents: nil)
        }
        return service.logFile
    }

    private func clearConflictingListenersIfNeeded(
        for service: ResolvedService,
        environment: [String: String],
        failIfNotCleared: Bool
    ) throws {
        guard let host = service.healthCheck.url.host, Self.isLocalHost(host) else { return }
        guard let port = Self.port(for: service.healthCheck.url) else { return }

        let currentPID = getpid()
        var pids = Self.listeningPIDs(on: port, environment: environment).filter { $0 != currentPID }
        guard !pids.isEmpty else { return }

        appendServiceLog(
            "Detected listener(s) on port \(port): \(pids.map(String.init).joined(separator: ", ")). Sending SIGTERM.",
            for: service
        )
        for pid in pids {
            _ = kill(pid, SIGTERM)
        }

        usleep(800_000)

        pids = Self.listeningPIDs(on: port, environment: environment).filter { $0 != currentPID }
        if !pids.isEmpty {
            appendServiceLog(
                "Listener(s) still active on port \(port): \(pids.map(String.init).joined(separator: ", ")). Sending SIGKILL.",
                for: service
            )
            for pid in pids {
                _ = kill(pid, SIGKILL)
            }
            usleep(400_000)
        }

        let remaining = Self.listeningPIDs(on: port, environment: environment).filter { $0 != currentPID }
        if !remaining.isEmpty {
            appendServiceLog(
                "Port \(port) still in use after cleanup attempt: \(remaining.map(String.init).joined(separator: ", "))",
                for: service
            )
            if failIfNotCleared {
                throw ServiceStartError.portStillInUse(port: port, pids: remaining)
            }
        } else {
            appendServiceLog("Port \(port) is clear.", for: service)
        }
    }

    private func appendServiceLog(_ line: String, for service: ResolvedService) {
        do {
            let logURL = try prepareLogFile(for: service)
            let handle = try FileHandle(forWritingTo: logURL)
            defer { try? handle.close() }
            handle.seekToEndOfFile()
            Self.writeLogLine("[control-center] \(line)", to: handle)
        } catch {
            // Best effort logging only.
        }
    }

    nonisolated private static func listeningPIDs(on port: Int, environment: [String: String]) -> [Int32] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        process.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        process.environment = environment

        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return []
        }

        let data = output.fileHandleForReading.readDataToEndOfFile()
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        return text
            .split(whereSeparator: \.isNewline)
            .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    nonisolated private static func isLocalHost(_ host: String) -> Bool {
        let normalized = host.lowercased()
        return normalized == "127.0.0.1"
            || normalized == "localhost"
            || normalized == "0.0.0.0"
            || normalized == "::1"
    }

    nonisolated private static func port(for url: URL) -> Int? {
        if let explicitPort = url.port {
            return explicitPort
        }
        switch url.scheme?.lowercased() {
        case "http":
            return 80
        case "https":
            return 443
        default:
            return nil
        }
    }

    private func makeEnvironment(for service: ResolvedService) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        for (key, value) in service.environment {
            env[key] = value
        }
        if service.id == "stable_audio",
           env["HF_TOKEN"]?.isEmpty != false,
           let token = StableAudioAuthKeychain.readToken(),
           !token.isEmpty {
            env["HF_TOKEN"] = token
        }
        return env
    }

    private func handleTermination(serviceID: String, process: Process) {
        guard let index = indexForService(serviceID) else { return }

        let expectedStop = services[index].processState == .stopping
        let status = process.terminationStatus
        services[index].pid = nil
        services[index].lastExitCode = status
        services[index].healthState = .unknown

        cleanupIO(for: serviceID)
        processes[serviceID] = nil
        healthTasks[serviceID]?.cancel()
        healthTasks[serviceID] = nil

        if expectedStop || status == 0 {
            services[index].processState = .stopped
            services[index].lastError = nil
        } else {
            services[index].processState = .failed
            services[index].lastError = "Exited with status \(status)"
            if services[index].service.restartOnCrash {
                restart(serviceID: serviceID)
            }
        }
    }

    private func cleanupIO(for serviceID: String) {
        if let pipe = outputPipes[serviceID] {
            pipe.fileHandleForReading.readabilityHandler = nil
        }
        outputPipes[serviceID] = nil

        if let handle = logHandles[serviceID] {
            try? handle.close()
        }
        logHandles[serviceID] = nil
    }

    private func startHealthMonitor(for service: ResolvedService) {
        healthTasks[service.id]?.cancel()
        healthTasks[service.id] = Task { [weak self] in
            while !Task.isCancelled {
                await self?.checkHealth(for: service)
                try? await Task.sleep(
                    nanoseconds: UInt64(service.healthCheck.intervalSeconds) * 1_000_000_000
                )
            }
        }
    }

    private func checkHealth(for service: ResolvedService) async {
        guard let index = indexForService(service.id) else { return }
        guard processes[service.id]?.isRunning == true else {
            services[index].healthState = .unknown
            return
        }

        var request = URLRequest(url: service.healthCheck.url)
        request.timeoutInterval = TimeInterval(service.healthCheck.timeoutSeconds)
        request.cachePolicy = .reloadIgnoringLocalCacheData

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                services[index].healthState = .unhealthy
                return
            }
            services[index].healthState = http.statusCode == service.healthCheck.expectedStatus ? .healthy : .unhealthy
        } catch {
            services[index].healthState = .unhealthy
        }
    }

    private func indexForService(_ id: String) -> Int? {
        services.firstIndex(where: { $0.id == id })
    }
}
