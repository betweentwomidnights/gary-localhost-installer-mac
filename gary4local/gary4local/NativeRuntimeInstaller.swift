import Foundation
import Darwin

nonisolated struct NativeRuntimeDefinition: Codable, Sendable {
    let version: String
    let packageURL: URL
    let sha256: String
    let signingTeam: String
    enum CodingKeys: String, CodingKey {
        case version, sha256
        case packageURL = "package_url"
        case signingTeam = "signing_team"
    }
}

nonisolated struct NativeRuntimeReceipt: Codable, Sendable {
    let version: String
    let sha256: String
    let installedAt: Date
    let recommendedEncoding: String
    let props: Data
    static func read(for service: ResolvedService) -> Self? {
        guard FileManager.default.isExecutableFile(atPath: service.executable.path),
              let data = try? Data(contentsOf: service.workingDirectory.appendingPathComponent("install.json")),
              let receipt = try? JSONDecoder().decode(Self.self, from: data),
              receipt.version == service.nativeRuntime?.version,
              receipt.sha256 == service.nativeRuntime?.sha256 else { return nil }
        return receipt
    }
}

nonisolated enum NativeRuntimeInstaller {
    // Output goes to private temporary files, avoiding subprocess pipe deadlocks.
    // All calls run off the UI thread and have a bounded timeout.
    static func run(_ executable: URL, _ arguments: [String], timeout: Double = 45) throws -> Data {
        let fm = FileManager.default
        let directory = fm.temporaryDirectory.appendingPathComponent("gary-native-probe-\(UUID().uuidString)")
        try fm.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: directory) }
        let out = directory.appendingPathComponent("stdout")
        let err = directory.appendingPathComponent("stderr")
        fm.createFile(atPath: out.path, contents: nil)
        fm.createFile(atPath: err.path, contents: nil)
        let stdout = try FileHandle(forWritingTo: out)
        let stderr = try FileHandle(forWritingTo: err)
        defer { try? stdout.close(); try? stderr.close() }
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        // Do not inherit debugger injection or credentials into release tools.
        process.environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": NSHomeDirectory()]
        try process.run()
        let deadline = DispatchWorkItem { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: deadline)
        process.waitUntilExit()
        deadline.cancel()
        let output = try Data(contentsOf: out)
        guard process.terminationStatus == 0 else {
            let detail = String(data: (try? Data(contentsOf: err)) ?? Data(), encoding: .utf8) ?? ""
            throw NativeServiceError.invalid("\(executable.lastPathComponent) failed (\(process.terminationStatus)): \(detail.suffix(2000))")
        }
        return output
    }

    static func install(service: ResolvedService,
                        report: @escaping @Sendable (String) -> Void) async throws -> NativeRuntimeReceipt {
        guard let config = service.nativeRuntime else {
            throw NativeServiceError.invalid("No native runtime configured.")
        }
        guard ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 13, minorVersion: 3, patchVersion: 0)) else {
            throw NativeServiceError.invalid("Yuey requires macOS 13.3 or newer.")
        }
        let fm = FileManager.default
        let parent = service.workingDirectory.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let stage = parent.appendingPathComponent(".install-\(UUID().uuidString)")
        try fm.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: stage) }
        let archive = stage.appendingPathComponent("runtime.zip")
        report("downloading signed Yuey \(config.version) runtime...")
        try await VerifiedNativeDownload.fetch(config.packageURL, to: archive, sha256: config.sha256) {
            report("downloading runtime: \(Int($0 * 100))%")
        }
        report("verifying signature and probing native backend...")
        let payload = stage.appendingPathComponent("payload")
        let receipt = try await Task.detached(priority: .userInitiated) {
            let listing = try run(URL(fileURLWithPath: "/usr/bin/unzip"), ["-Z1", archive.path])
            let entries = (String(data: listing, encoding: .utf8) ?? "").split(separator: "\n")
            guard !entries.isEmpty, entries.allSatisfy({
                !$0.hasPrefix("/") && !$0.split(separator: "/").contains("..")
            }) else { throw NativeServiceError.invalid("Unsafe runtime archive paths.") }
            try fm.createDirectory(at: payload, withIntermediateDirectories: false)
            _ = try run(URL(fileURLWithPath: "/usr/bin/unzip"), ["-q", archive.path, "-d", payload.path])
            let server = payload.appendingPathComponent(service.executable.lastPathComponent)
            // CLI binaries are not app bundles: spctl --type execute rejects them
            // even when properly notarized. Verify signed code plus the pinned team.
            let binaries = try fm.contentsOfDirectory(at: payload, includingPropertiesForKeys: [.isSymbolicLinkKey])
            for file in binaries {
                let values = try file.resourceValues(forKeys: [.isSymbolicLinkKey])
                guard values.isSymbolicLink != true else { throw NativeServiceError.invalid("Unexpected runtime symlink.") }
                if file.pathExtension == "dylib" || file.lastPathComponent == server.lastPathComponent {
                    _ = try run(URL(fileURLWithPath: "/usr/bin/codesign"), ["--verify", "--strict",
                        "-R=anchor apple generic and certificate leaf[subject.OU] = \"\(config.signingTeam)\"", file.path])
                }
            }
            let version = String(data: try run(server, ["--version"]), encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard version == config.version.trimmingCharacters(in: CharacterSet(charactersIn: "v")) else {
                throw NativeServiceError.invalid("Runtime version does not match the pinned release.")
            }
            let props = try run(server, ["--props", "--device", YueySettings.automaticDevice,
                "--models-dir", YueyModelCatalog.modelDirectory(for: service).path, "--force-unload"])
            guard let json = try JSONSerialization.jsonObject(with: props) as? [String: Any],
                  json["success"] as? Bool == true, json["service"] as? String == "yue2" else {
                throw NativeServiceError.invalid("Native runtime backend probe failed.")
            }
            let hardware = json["hardware"] as? [String: Any]
            return NativeRuntimeReceipt(version: config.version, sha256: config.sha256, installedAt: Date(),
                recommendedEncoding: hardware?["recommended_encoding"] as? String ?? "Q4_K_M", props: props)
        }.value
        try Task.checkCancellation()
        try JSONEncoder().encode(receipt).write(to: payload.appendingPathComponent("install.json"), options: .atomic)
        // Keep a recoverable previous runtime; never mutate the application's bundle.
        var backup: URL?
        if fm.fileExists(atPath: service.workingDirectory.path) {
            let previous = parent.appendingPathComponent("backup-\(config.version)-\(UUID().uuidString)")
            try fm.moveItem(at: service.workingDirectory, to: previous)
            backup = previous
        }
        do { try fm.moveItem(at: payload, to: service.workingDirectory) }
        catch {
            if let backup { try? fm.moveItem(at: backup, to: service.workingDirectory) }
            throw error
        }
        report("Yuey \(config.version) installed and verified. Download models next.")
        return receipt
    }
}
