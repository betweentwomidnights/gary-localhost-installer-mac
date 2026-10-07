import Foundation

nonisolated struct YueyModelFile: Identifiable, Sendable {
    let name: String
    let bytes: Int64
    let sha256: String
    var id: String { name }
}

nonisolated enum YueyModelCatalog {
    static let repository = "thepatch/YuE2-3B-GGUF"
    static let revision = "0d62af46c4945fe04ea241fa83db0c53f80f6cfc"
    static let tiers = ["Q4_K_M", "Q8_0", "BF16"]
    static let shared: [YueyModelFile] = [
        .init(name: "yue2-vae-v1.0-F16.gguf", bytes: 265190208, sha256: "b6d8d30e69ce997b50f25dbbdd610e9ac7e118bbda91af336550862a6d1aab1f"),
        .init(name: "yue2-qwen.tiktoken", bytes: 2561218, sha256: "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"),
        .init(name: "sheetsage2-mert2-0.7B-v1.0-F16.gguf", bytes: 1355653664, sha256: "29902ab4c24a5dd4571673210af4c93f9e70f3ad9fcd18dc5b923a839d4bcad0"),
        .init(name: "yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf", bytes: 1352153056, sha256: "e0005d6ea838e444c83561609624f85a0c0f6fc703bade2e477eb034a327a0cc"),
        .init(name: "yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf", bytes: 139491200, sha256: "45a6ec461c968916bdd4d87269cff1324bb925e7ead35290524dfa82bf3f9c23"),
        .init(name: "yue2-realaudio-nar-v9-v1.0-F16-LoRA.gguf", bytes: 70291232, sha256: "4b7bad23fd1e52b9a9540dde568b6226afc37229b756a4c2024f551be8a2557c")
    ]
    static let generation: [YueyModelFile] = [
        .init(name: "yue2-3.6B-v1.0-Q4_K_M.gguf", bytes: 2529174976, sha256: "cca9ab2fc8efca09ddc04d68da63ae6c6e535848bb3b39dc52cb820857281ba9"),
        .init(name: "yue2-3.6B-v1.0-Q8_0.gguf", bytes: 3909744064, sha256: "db3fa40df040288a41802f2af6df62e63a7e985b026ea60d1e0cdfc36469ad79"),
        .init(name: "yue2-3.6B-v1.0-BF16.gguf", bytes: 7261418848, sha256: "9e1e3bb89dce35703345cd39c57a43f996d62e5ca87afb6784ff5ecfcad27a55")
    ]
    static func file(for tier: String) -> YueyModelFile? {
        generation.first { $0.name == "yue2-3.6B-v1.0-\(tier).gguf" }
    }
    static func modelDirectory(for service: ResolvedService) -> URL {
        URL(fileURLWithPath: service.environment["YUE2_MODELS_DIR"] ??
            service.workingDirectory.appendingPathComponent("models").path, isDirectory: true)
    }
    static func installed(_ file: YueyModelFile, in directory: URL) -> Bool {
        let url = directory.appendingPathComponent(file.name)
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              values.isRegularFile == true, values.isSymbolicLink != true else { return false }
        return Int64(values.fileSize ?? 0) == file.bytes
    }
    static func installedTiers(in directory: URL) -> [String] {
        tiers.filter { tier in file(for: tier).map { installed($0, in: directory) } ?? false }
    }
    static func chooseEncoding(installed: [String], chosen: String, recommended: String) -> String? {
        if installed.contains(chosen) { return chosen }
        if installed.contains(recommended) { return recommended }
        return tiers.first { installed.contains($0) }
    }
    static func defaultTier(recommended: String) -> String {
        switch recommended {
        case "BF16", "F16", "F32": return "BF16"
        case "Q8_0": return "Q8_0"
        default: return "Q4_K_M"
        }
    }
    static func missing(in directory: URL) -> [String] {
        var missing = shared.filter { !installed($0, in: directory) }.map(\.name)
        if installedTiers(in: directory).isEmpty { missing.append("one generation tier (Q4_K_M, Q8_0 or BF16)") }
        return missing
    }
}

nonisolated struct YueySettings: Codable, Equatable, Sendable {
    var device = "auto"
    var encoding = "auto"
    var instrumentalMethod = "transfer"
    var useInstrumentalAdapter = true
    var naturalMaxSeconds = 96
    static let defaultsKey = "yueyNativeSettingsV1"
    static func load(from defaults: UserDefaults = .standard) -> Self {
        guard let data = defaults.data(forKey: defaultsKey),
              let saved = try? JSONDecoder().decode(Self.self, from: data) else { return .init() }
        return saved.normalized
    }
    var normalized: Self {
        var result = self
        if !["auto", "mtl", "cpu"].contains(device) { result.device = "auto" }
        if !(["auto"] + YueyModelCatalog.tiers).contains(encoding) { result.encoding = "auto" }
        if !["transfer", "rest"].contains(instrumentalMethod) { result.instrumentalMethod = "transfer" }
        result.naturalMaxSeconds = max(30, min(600, naturalMaxSeconds))
        return result
    }
    static var automaticDevice: String {
        #if arch(arm64)
        return "mtl"
        #else
        return "cpu"
        #endif
    }
    func environment(installed: [String], recommended: String) -> [String: String] {
        let clean = normalized
        return [
            "YUE2_DEVICE": clean.device == "auto" ? Self.automaticDevice : clean.device,
            "YUE2_ENCODING": YueyModelCatalog.chooseEncoding(installed: installed, chosen: clean.encoding, recommended: recommended) ?? "auto",
            "YUE2_INSTRUMENTAL_METHOD": clean.instrumentalMethod,
            "YUE2_USE_INSTRUMENTAL_ADAPTER": clean.useInstrumentalAdapter ? "1" : "0",
            "YUE2_NATURAL_MAX_SECONDS": String(clean.naturalMaxSeconds)
        ]
    }
}
