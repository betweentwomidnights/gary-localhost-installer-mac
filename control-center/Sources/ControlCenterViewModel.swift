import Foundation
import SwiftUI
import Combine
import AppKit

@MainActor
final class ControlCenterViewModel: ObservableObject {
    @Published var manager: ServiceManager?
    @Published var startupError: String?
    @Published var manifestPath: String = ""
    @Published var selectedServiceID: String?
    @Published var selectedLogText: String = ""
    @Published var stableAudioTokenInput: String = ""
    @Published var stableAudioTokenConfigured: Bool = false
    @Published var stableAudioTokenStatus: String = ""
    @Published var stableAudioStep2ScreenshotPath: String?

    private var logRefreshTask: Task<Void, Never>?
    private var cancellables = Set<AnyCancellable>()

    init() {
        observeApplicationTermination()
        refreshStableAudioTokenState()
        loadManifest()
    }

    deinit {
        logRefreshTask?.cancel()
    }

    func loadManifest() {
        let defaultURL = ManifestLoader.defaultManifestURL()
        manifestPath = defaultURL.path
        reloadHFScreenshots()
        refreshStableAudioTokenState()

        do {
            let manifest = try ManifestLoader.load(from: defaultURL)
            let manager = ServiceManager(manifest: manifest)
            self.manager = manager
            startupError = nil
            selectedServiceID = manager.services.first?.id
            selectedLogText = selectedServiceID.map { manager.readLogTail(serviceID: $0) } ?? ""
            manager.startAutoStartServices()
            startLogRefreshLoop()
        } catch {
            self.manager = nil
            startupError = error.localizedDescription
        }
    }

    func selectService(_ serviceID: String) {
        selectedServiceID = serviceID
        refreshLog()
    }

    func refreshLog() {
        guard let manager, let selectedServiceID else {
            selectedLogText = ""
            return
        }
        selectedLogText = manager.readLogTail(serviceID: selectedServiceID)
    }

    func saveStableAudioToken() {
        let token = stableAudioTokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            stableAudioTokenStatus = "Paste your Hugging Face token first."
            return
        }

        do {
            try StableAudioAuthKeychain.saveToken(token)
            stableAudioTokenInput = ""
            stableAudioTokenConfigured = true
            stableAudioTokenStatus = "Token saved in Keychain."
        } catch {
            stableAudioTokenStatus = error.localizedDescription
        }
    }

    func clearStableAudioToken() {
        do {
            try StableAudioAuthKeychain.deleteToken()
            stableAudioTokenConfigured = false
            stableAudioTokenInput = ""
            stableAudioTokenStatus = "Saved token removed."
        } catch {
            stableAudioTokenStatus = error.localizedDescription
        }
    }

    func refreshStableAudioTokenState() {
        stableAudioTokenConfigured = StableAudioAuthKeychain.readToken()?.isEmpty == false
        if stableAudioTokenConfigured {
            if stableAudioTokenStatus.isEmpty {
                stableAudioTokenStatus = "Token already saved in Keychain."
            }
        } else if stableAudioTokenStatus == "Token already saved in Keychain." {
            stableAudioTokenStatus = ""
        }
    }

    private func reloadHFScreenshots() {
        guard !manifestPath.isEmpty else {
            stableAudioStep2ScreenshotPath = nil
            return
        }

        guard let screenshotDirectory = findScreenshotDirectory(startingAt: manifestPath) else {
            stableAudioStep2ScreenshotPath = nil
            return
        }

        let fileManager = FileManager.default
        let allowedExtensions = Set(["png", "jpg", "jpeg", "webp"])

        let paths = (try? fileManager.contentsOfDirectory(
            at: screenshotDirectory,
            includingPropertiesForKeys: nil
        ))?.filter { url in
            allowedExtensions.contains(url.pathExtension.lowercased())
        }
        .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        .map(\.path) ?? []

        stableAudioStep2ScreenshotPath = preferredStep2Screenshot(from: paths)
    }

    private func preferredStep2Screenshot(from paths: [String]) -> String? {
        if let preferred = paths.first(where: { $0.contains("6.51.30") }) {
            return preferred
        }
        return paths.first
    }

    private func findScreenshotDirectory(startingAt manifestPath: String) -> URL? {
        var cursor = URL(fileURLWithPath: manifestPath).deletingLastPathComponent()
        let fileManager = FileManager.default

        while cursor.path != "/" {
            let candidate = cursor.appendingPathComponent("hf-screenshots")
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
               isDirectory.boolValue {
                return candidate
            }
            cursor.deleteLastPathComponent()
        }
        return nil
    }

    private func startLogRefreshLoop() {
        logRefreshTask?.cancel()
        logRefreshTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                await MainActor.run {
                    self?.refreshLog()
                }
            }
        }
    }

    private func observeApplicationTermination() {
        NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                self?.logRefreshTask?.cancel()
                self?.manager?.shutdownForApplicationTermination()
            }
            .store(in: &cancellables)
    }
}
