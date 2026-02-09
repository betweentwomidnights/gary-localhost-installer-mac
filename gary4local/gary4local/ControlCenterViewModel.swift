import Foundation
import SwiftUI
import Combine

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

    init() {
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

        stableAudioTokenStatus = "Saving token..."
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                try StableAudioAuthKeychain.saveToken(token)
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.stableAudioTokenInput = ""
                    self.applyStableAudioTokenState(configured: true)
                    self.stableAudioTokenStatus = "Token saved in Keychain."
                }
            } catch {
                DispatchQueue.main.async {
                    self?.stableAudioTokenStatus = error.localizedDescription
                }
            }
        }
    }

    func clearStableAudioToken() {
        stableAudioTokenStatus = "Removing token..."
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                try StableAudioAuthKeychain.deleteToken()
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.stableAudioTokenInput = ""
                    self.applyStableAudioTokenState(configured: false)
                    self.stableAudioTokenStatus = "Saved token removed."
                }
            } catch {
                DispatchQueue.main.async {
                    self?.stableAudioTokenStatus = error.localizedDescription
                }
            }
        }
    }

    func refreshStableAudioTokenState() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let configured = StableAudioAuthKeychain.readToken()?.isEmpty == false
            DispatchQueue.main.async {
                self?.applyStableAudioTokenState(configured: configured)
            }
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

    private func applyStableAudioTokenState(configured: Bool) {
        stableAudioTokenConfigured = configured
        if configured {
            if stableAudioTokenStatus.isEmpty {
                stableAudioTokenStatus = "Token already saved in Keychain."
            }
        } else if stableAudioTokenStatus == "Token already saved in Keychain." {
            stableAudioTokenStatus = ""
        }
    }
}
