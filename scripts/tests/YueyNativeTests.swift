// Foundation-only tests of the production manifest, catalog, verifier and installer.
// Compile with ManifestModels.swift, ManifestLoader.swift, YueyModelCatalog.swift,
// NativeRuntimeInstaller.swift and VerifiedNativeDownload.swift.
import Foundation

final class NativeTestProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var fractions: [Double] = []
    func record(_ value: Double) { lock.lock(); fractions.append(value); lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return fractions.count }
}

@main
struct YueyNativeTests {
    static func main() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("yuey-tests-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: root) }
        let manifest = try ManifestLoader.load(from: URL(fileURLWithPath: "control-center/manifest/services.production.json"))
        let service = manifest.services.first { $0.id == "yuey" }!
        precondition(service.bootstrap == nil && service.nativeRuntime?.version == "v0.2.2")
        precondition(service.arguments.contains("--force-unload") && !service.arguments.contains("--keep-models"))
        precondition(service.healthCheck.url.absoluteString == "http://127.0.0.1:8007/health")
        precondition(service.environment["YUE2_MODELS_DIR"]!.hasSuffix("/models/yuey"))
        precondition(manifest.services.count == 7)
        let dev = try ManifestLoader.load(from: URL(fileURLWithPath: "control-center/manifest/services.dev.json"))
        precondition(dev.services.first { $0.id == "yuey" }!.nativeRuntime?.sha256 == service.nativeRuntime?.sha256)
        precondition(YueyModelCatalog.shared.count == 6 && YueyModelCatalog.generation.count == 3)
        precondition(Set((YueyModelCatalog.shared + YueyModelCatalog.generation).map(\.name)).count == 9)
        precondition((YueyModelCatalog.shared + YueyModelCatalog.generation).allSatisfy { $0.sha256.count == 64 && $0.bytes > 0 })
        precondition(YueyModelCatalog.missing(in: root).count == 7)
        precondition(YueyModelCatalog.defaultTier(recommended: "Q5_K_M") == "Q4_K_M")
        precondition(YueyModelCatalog.defaultTier(recommended: "BF16") == "BF16")
        precondition(YueyModelCatalog.chooseEncoding(installed: ["Q4_K_M", "BF16"], chosen: "Q4_K_M", recommended: "BF16") == "Q4_K_M")
        precondition(YueyModelCatalog.chooseEncoding(installed: ["Q4_K_M"], chosen: "BF16", recommended: "BF16") == "Q4_K_M")
        precondition(YueyModelCatalog.chooseEncoding(installed: [], chosen: "auto", recommended: "BF16") == nil)
        let tiny = root.appendingPathComponent("tiny.gguf")
        try Data("abc".utf8).write(to: tiny)
        let tinyHash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        let actualTinyHash = try VerifiedNativeDownload.sha256(tiny)
        precondition(actualTinyHash == tinyHash)
        let file = YueyModelFile(name: "tiny.gguf", bytes: 3, sha256: tinyHash)
        precondition(YueyModelCatalog.installed(file, in: root))
        precondition(!YueyModelCatalog.installed(.init(name: "tiny.gguf", bytes: 4, sha256: tinyHash), in: root))
        let link = root.appendingPathComponent("symlink.gguf")
        try fm.createSymbolicLink(at: link, withDestinationURL: tiny)
        precondition(!YueyModelCatalog.installed(.init(name: "symlink.gguf", bytes: 3, sha256: tinyHash), in: root))
        var settings = YueySettings()
        precondition(settings.naturalMaxSeconds == 96 && settings.instrumentalMethod == "transfer" && settings.useInstrumentalAdapter)
        settings.device = "cuda"; settings.encoding = "Q5_K_M"; settings.naturalMaxSeconds = 999; settings.instrumentalMethod = "bad"
        precondition(settings.normalized.device == "auto" && settings.normalized.encoding == "auto")
        precondition(settings.normalized.naturalMaxSeconds == 600 && settings.normalized.instrumentalMethod == "transfer")
        settings = .init(); settings.device = "cpu"; settings.naturalMaxSeconds = 1
        let env = settings.environment(installed: ["Q4_K_M"], recommended: "BF16")
        precondition(env["YUE2_DEVICE"] == "cpu" && env["YUE2_ENCODING"] == "Q4_K_M" && env["YUE2_NATURAL_MAX_SECONDS"] == "30")
        let suiteName = "yuey-test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set(try JSONEncoder().encode(settings), forKey: YueySettings.defaultsKey)
        precondition(YueySettings.load(from: defaults).naturalMaxSeconds == 30)
        print("Yuey catalog, manifest, settings and file-verifier fixtures passed.")

        if CommandLine.arguments.contains("--network") {
            let tokenizer = YueyModelCatalog.shared.first { $0.name == "yue2-qwen.tiktoken" }!
            let url = URL(string: "https://huggingface.co/\(YueyModelCatalog.repository)/resolve/\(YueyModelCatalog.revision)/\(tokenizer.name)")!
            let downloaded = root.appendingPathComponent(tokenizer.name)
            let progress = NativeTestProgress()
            try await VerifiedNativeDownload.fetch(url, to: downloaded, sha256: tokenizer.sha256,
                expectedBytes: tokenizer.bytes) { progress.record($0) }
            precondition(YueyModelCatalog.installed(tokenizer, in: root))
            precondition(progress.count > 0, "download did not report UI progress")
            let rejected = root.appendingPathComponent("bad-checksum")
            do {
                try await VerifiedNativeDownload.fetch(url, to: rejected, sha256: String(repeating: "0", count: 64)) { _ in }
                preconditionFailure("bad checksum was accepted")
            } catch {
                precondition(error.localizedDescription.contains("SHA-256 mismatch"))
                precondition(!fm.fileExists(atPath: rejected.path))
            }
            print("Pinned tokenizer download, UI progress and corrupt-checksum rejection passed.")
            let cancelledFile = root.appendingPathComponent("cancelled-download")
            let cancelled = Task {
                try await VerifiedNativeDownload.fetch(url, to: cancelledFile, sha256: tokenizer.sha256) { _ in }
            }
            cancelled.cancel()
            do { try await cancelled.value; preconditionFailure("cancelled download installed") }
            catch { precondition(!fm.fileExists(atPath: cancelledFile.path)) }
            print("Cancelled download does not install an incomplete file.")
        }
        if CommandLine.arguments.contains("--install-runtime") {
            let receipt = try await NativeRuntimeInstaller.install(service: service) { print($0) }
            precondition(receipt.version == "v0.2.2")
            precondition(NativeRuntimeReceipt.read(for: service)?.sha256 == service.nativeRuntime?.sha256)
            print("Production native-runtime download, signature check, version and Metal probe passed.")
        }
        if CommandLine.arguments.contains("--verify-local-models") {
            let source = URL(fileURLWithPath: "/Users/karenjessen/yuey.cpp/models")
            for file in YueyModelCatalog.shared + [YueyModelCatalog.file(for: "Q4_K_M")!] {
                let hash = try VerifiedNativeDownload.sha256(source.appendingPathComponent(file.name))
                precondition(hash == file.sha256, "local file failed verification: \(file.name)")
                print("verified local \(file.name)")
            }
        }
    }
}
