import Foundation

@main
struct SA3ModelDownloadTests {
    static func main() {
        precondition(SA3PredownloadTarget.required.requiresToken)
        precondition(!SA3PredownloadTarget.decoder.requiresToken)
        precondition(SA3PredownloadTarget.required.payload == ["target_type": "required"])
        precondition(SA3PredownloadTarget.decoder.payload == ["target_type": "decoder"])
        precondition(SA3PredownloadTarget.decoder.helperCommand == "download-decoder")
        precondition(SA3PredownloadTarget.required.helperCommand == "download-required")
        precondition(SA3PredownloadTarget.decoder.label == "SAME-L decoder squeak fix")
        print("SA3 API/CLI target identities and optional-token policy passed.")
    }
}
