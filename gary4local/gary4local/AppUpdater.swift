import SwiftUI
import Combine
import Sparkle

private final class GaryAppUpdaterDelegate: NSObject, SPUUpdaterDelegate {
    private let feedURLOverride: String?

    init(feedURLOverride: String?) {
        self.feedURLOverride = feedURLOverride
    }

    func feedURLString(for updater: SPUUpdater) -> String? {
        feedURLOverride
    }

    func updater(_ updater: SPUUpdater, didFinishLoading appcast: SUAppcast) {
        print("[sparkle] loaded appcast from \(updater.feedURL?.absoluteString ?? "unknown")")
    }

    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        print(
            "[sparkle] found update shortVersion=\(item.displayVersionString) buildVersion=\(item.versionString)"
        )
    }

    func updaterDidNotFindUpdate(_ updater: SPUUpdater, error: Error) {
        print("[sparkle] no update found: \(error.localizedDescription)")
    }
}

@MainActor
final class GaryAppUpdater {
    let updaterController: SPUStandardUpdaterController

    private let updaterDelegate: GaryAppUpdaterDelegate

    init(processInfo: ProcessInfo = .processInfo) {
        updaterDelegate = GaryAppUpdaterDelegate(
            feedURLOverride: GaryAppUpdater.resolveFeedURLOverride(processInfo: processInfo)
        )
        updaterController = SPUStandardUpdaterController(
            startingUpdater: false,
            updaterDelegate: updaterDelegate,
            userDriverDelegate: nil
        )
        updaterController.startUpdater()
    }

    private static func resolveFeedURLOverride(processInfo: ProcessInfo) -> String? {
        guard let rawValue = processInfo.environment["GARY4LOCAL_SPARKLE_FEED_URL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !rawValue.isEmpty else {
            return nil
        }

        guard let url = URL(string: rawValue), url.scheme != nil else {
            print("[sparkle] ignoring invalid feed override: \(rawValue)")
            return nil
        }

        print("[sparkle] using feed override: \(url.absoluteString)")
        return url.absoluteString
    }
}

@MainActor
final class CheckForUpdatesViewModel: ObservableObject {
    @Published private(set) var canCheckForUpdates: Bool

    private var canCheckForUpdatesObservation: AnyCancellable?

    init(updater: SPUUpdater) {
        canCheckForUpdates = updater.canCheckForUpdates
        canCheckForUpdatesObservation = updater
            .publisher(for: \.canCheckForUpdates, options: [.initial, .new])
            .receive(on: RunLoop.main)
            .sink { [weak self] canCheckForUpdates in
                self?.canCheckForUpdates = canCheckForUpdates
            }
    }
}

struct GaryUpdateCommands: Commands {
    @ObservedObject private var viewModel: CheckForUpdatesViewModel

    private let updaterController: SPUStandardUpdaterController

    init(updaterController: SPUStandardUpdaterController) {
        self.updaterController = updaterController
        _viewModel = ObservedObject(
            wrappedValue: CheckForUpdatesViewModel(updater: updaterController.updater)
        )
    }

    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button("check for updates...") {
                updaterController.checkForUpdates(nil)
            }
            .disabled(!viewModel.canCheckForUpdates)
        }
    }
}
