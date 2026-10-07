// Compile alongside production ServiceManager, manifest, native download/catalog/
// installer and StableAudioAuth sources. This fixture supplies only the unrelated
// SA3 settings type so the real Yuey manager path can run without a SwiftUI app.
import Foundation

struct SA3RuntimeSettings: Equatable {
    var peakNormalizeDb = "2", limiterCeilingDb = "-0.3", latentRescale = "1", latentShift = "0"
    var latentTargetStd = "", continuationTailPad = "6", continuationSpliceXfade = "0.03", continuationMaskOverlap = "0.2"
    var continuationSpliceSource = true, continuationSpliceGainMatch = true, useDecoderLora = false, useFP32DiT = true
    static let fallbackDefaults = Self()
    var normalized: Self { self }
}

@main
struct YueyServiceManagerSmoke {
    @MainActor
    static func main() async throws {
        let manifest = try ManifestLoader.load(from: URL(fileURLWithPath: "control-center/manifest/services.production.json"))
        let service = manifest.services.first { $0.id == "yuey" }!
        let temporaryModels = FileManager.default.temporaryDirectory.appendingPathComponent("yuey-missing-models-\(UUID().uuidString)")
        var emptyEnvironment = service.environment
        emptyEnvironment["YUE2_MODELS_DIR"] = temporaryModels.path
        let missingModelsService = ResolvedService(id: service.id, name: service.name, workingDirectory: service.workingDirectory,
            executable: service.executable, arguments: service.arguments, environment: emptyEnvironment,
            logFile: service.logFile, healthCheck: service.healthCheck, bootstrap: nil, autoStart: false,
            restartOnCrash: false, gracefulShutdownSeconds: service.gracefulShutdownSeconds, nativeRuntime: service.nativeRuntime)
        let guardManifest = ResolvedManifest(schemaVersion: 1, appName: "fixture", defaultLogDirectory: manifest.defaultLogDirectory,
            services: [missingModelsService])
        let guardManager = ServiceManager(manifest: guardManifest)
        guardManager.start(serviceID: "yuey")
        precondition(guardManager.services[0].processState == .failed)
        precondition(guardManager.services[0].lastError?.contains("Download Yuey models first") == true)
        var noRuntimeService = missingModelsService
        noRuntimeService.nativeRuntime = NativeRuntimeDefinition(version: "v0.0.0", packageURL: service.nativeRuntime!.packageURL,
            sha256: service.nativeRuntime!.sha256, signingTeam: service.nativeRuntime!.signingTeam)
        let noRuntimeManifest = ResolvedManifest(schemaVersion: 1, appName: "fixture", defaultLogDirectory: manifest.defaultLogDirectory,
            services: [noRuntimeService])
        let noRuntimeManager = ServiceManager(manifest: noRuntimeManifest)
        noRuntimeManager.start(serviceID: "yuey")
        precondition(noRuntimeManager.services[0].lastError?.contains("Install the verified Yuey runtime first") == true)
        let onlyYuey = ResolvedManifest(schemaVersion: manifest.schemaVersion, appName: manifest.appName,
            defaultLogDirectory: manifest.defaultLogDirectory, services: [service])
        let manager = ServiceManager(manifest: onlyYuey)
        defer { manager.shutdownForApplicationTermination() }
        manager.setNativeModelOperationInProgress(true)
        manager.start(serviceID: "yuey")
        precondition(manager.services[0].lastError?.contains("Finish the Yuey model download") == true)
        manager.setNativeModelOperationInProgress(false)
        manager.start(serviceID: "yuey")
        precondition(manager.services[0].processState == .running, manager.services[0].lastError ?? "launch failed")
        func api(_ path: String, body: [String: Any]? = nil) async throws -> [String: Any] {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:8007\(path)")!)
            request.timeoutInterval = 10
            if let body {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw NativeServiceError.invalid("API failed for \(path): \(String(data: data, encoding: .utf8) ?? "")")
            }
            return json
        }
        var ready = false
        for _ in 0..<120 {
            if let health = try? await api("/health"), health["service"] as? String == "yue2" { ready = true; break }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        precondition(ready, "health never became ready")
        let props = try await api("/props")
        let defaults = props["defaults"] as! [String: Any]
        precondition(defaults["force_unload"] as? Bool == true)
        precondition(defaults["natural_max_seconds"] as? Double == 96)
        precondition(defaults["instrumental_method"] as? String == "transfer")
        let initial = try await api("/health")
        precondition(initial["encoding"] as? String == "Q4_K_M")
        precondition((initial["generation"] as? [String: Any])?["loaded"] as? Bool == false)
        _ = try await api("/prompts")
        _ = try await api("/loras")
        print("Managed Yuey launch, health, defaults, Q4 tier fallback, prompts and adapter catalog passed.")
        fflush(stdout)
        if CommandLine.arguments.contains("--generation") {
            // Eight bars avoid relying on the known very-short-score budget bug.
            // An explicit tiny token budget and one ODE step exercise the plumbing,
            // not musical quality. Deliberately request keep_models to test override.
            let notes = Array(repeating: "C8E8G16|", count: 8).joined()
            let backing = Array(repeating: "C,32|", count: 8).joined()
            let abc = "X:1\nM:4/4\nL:1/32\nQ:1/4=120\nV: Vocal clef=treble\nV: Ins clef=treble\nK:C\n% verse\nV: Vocal\n\(notes)\nV: Ins\n\(backing)\n"
            let job = try await api("/generate", body: ["style": "acoustic, intimate", "lyrics": "[Verse]\nA quiet line",
                "abc": abc, "symbolic_mode": "full", "seed": 831001, "semantic_max_tokens": 32,
                "semantic_min_tokens": 2, "ode_steps": 1, "keep_models": true, "audio_format": "wav"])
            let id = job["session_id"] as! String
            var completed: [String: Any]?
            for _ in 0..<360 {
                let status = try await api("/poll_status/\(id)")
                if status["status"] as? String == "failed" {
                    throw NativeServiceError.invalid("generation failed: \(status["error"] ?? "unknown")")
                }
                if status["status"] as? String == "completed" { completed = status; break }
                try await Task.sleep(nanoseconds: 500_000_000)
            }
            guard let completed, let audio = completed["audio_data"] as? String,
                  let wav = Data(base64Encoded: audio), wav.count > 44 else {
                _ = try? await api("/cancel/\(id)", body: [:])
                throw NativeServiceError.invalid("generation did not return audio within the smoke-test deadline")
            }
            let output = URL(fileURLWithPath: "/tmp/yuey-mac-api-smoke.wav")
            try wav.write(to: output)
            var unloaded = false
            for _ in 0..<40 {
                let health = try await api("/health")
                if health["busy"] as? Bool == false,
                   (health["generation"] as? [String: Any])?["loaded"] as? Bool == false {
                    unloaded = true; break
                }
                try await Task.sleep(nanoseconds: 250_000_000)
            }
            precondition(unloaded, "models stayed loaded despite --force-unload")
            _ = try await api("/poll_status/\(id)?consume=1")
            print("Managed Metal/Q4 generation returned \(wav.count) WAV bytes and unloaded despite keep_models=true.")
            let cancelledJob = try await api("/generate", body: ["style": "acoustic", "abc": abc,
                "symbolic_mode": "full", "seed": 831002, "semantic_max_tokens": 2048, "keep_models": true])
            let cancelledID = cancelledJob["session_id"] as! String
            _ = try await api("/cancel/\(cancelledID)", body: [:])
            var cancelled = false
            for _ in 0..<120 {
                let status = try await api("/poll_status/\(cancelledID)")
                if status["cancelled"] as? Bool == true { cancelled = true; break }
                try await Task.sleep(nanoseconds: 250_000_000)
            }
            precondition(cancelled, "API cancellation was not acknowledged")
            let afterCancel = try await api("/health")
            precondition((afterCancel["generation"] as? [String: Any])?["loaded"] as? Bool == false)
            print("API cancellation acknowledged; generator remains unloaded.")
        }
        manager.stop(serviceID: "yuey")
        for _ in 0..<80 {
            if manager.services[0].processState == .stopped { break }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        precondition(manager.services[0].processState == .stopped, "managed service did not stop")
        print("Managed stop passed; port 8007 is released.")
    }
}
