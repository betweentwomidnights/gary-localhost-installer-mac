import Foundation

enum GaryAppReleaseInfo {
    static let recommendedGary4JuceVersion = "v3.0.3-mac"
    static let recommendedGary4JuceReleaseURL = URL(
        string: "https://github.com/betweentwomidnights/gary4juce/releases"
    )!

    static var shortVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
    }

    static var buildNumber: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0"
    }

    static var releaseLabel: String {
        "v\(shortVersion)"
    }

    static var detailedVersionLabel: String {
        "\(releaseLabel) (build \(buildNumber))"
    }
}
