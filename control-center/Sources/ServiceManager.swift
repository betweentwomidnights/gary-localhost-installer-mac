import Foundation
import Darwin

@MainActor
final class ServiceManager: ObservableObject {
    @Published private(set) var services: [ServiceRuntime]

    private let manifest: ResolvedManifest
    private var processes: [String: Process] = [:]
    private var outputPipes: [String: Pipe] = [:]
    private var logHandles: [String: FileHandle] = [:]
    private var healthTasks: [String: Task<Void, Never>] = [:]

    init(manifest: ResolvedManifest) {
        self.manifest = manifest
        self.services = manifest.services.map { ServiceRuntime(service: $0) }
    }

    deinit {
        for (_, task) in healthTasks {
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

        let service = services[index].service
        services[index].processState = .starting
        services[index].lastError = nil
        services[index].healthState = .unknown

        do {
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
            process.environment = makeEnvironment(for: service)
            process.terminationHandler = { [weak self] terminated in
                Task { @MainActor in
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

    func readLogTail(serviceID: String, maxLines: Int = 250) -> String {
        guard let service = services.first(where: { $0.id == serviceID })?.service else {
            return ""
        }
        guard let data = try? Data(contentsOf: service.logFile),
              let text = String(data: data, encoding: .utf8) else {
            return ""
        }

        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        let tail = lines.suffix(maxLines)
        return tail.joined(separator: "\n")
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

    private func makeEnvironment(for service: ResolvedService) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        for (key, value) in service.environment {
            env[key] = value
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
