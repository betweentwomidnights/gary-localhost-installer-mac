import Foundation

/// Shared request/CLI identity keeps online and stopped-service downloads aligned.
nonisolated enum SA3PredownloadTarget: Sendable {
    case required
    case decoder

    static let decoderRepository = "thepatch/same-l-decoder-lora"
    var label: String { self == .required ? "required sa3 models" : "SAME-L decoder squeak fix" }
    var requiresToken: Bool { self == .required }
    var helperCommand: String { self == .required ? "download-required" : "download-decoder" }
    var payload: [String: String] { ["target_type": self == .required ? "required" : "decoder"] }
}
