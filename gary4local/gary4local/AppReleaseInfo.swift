import Foundation

enum GaryAppReleaseInfo {
    static let recommendedGary4JuceVersion = "v4.0.7-mac"
    // Derived so the link and the displayed version cannot drift apart.
    static let recommendedGary4JuceReleaseURL = URL(
        string: "https://github.com/betweentwomidnights/gary4juce/releases/tag/"
            + recommendedGary4JuceVersion
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
