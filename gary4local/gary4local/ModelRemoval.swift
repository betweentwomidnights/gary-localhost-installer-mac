import Combine
import Foundation

/// Removing a downloaded model, with the checks that make it safe to offer.
///
/// The dangerous part is not the delete itself -- it is deleting out from under
/// something. macOS keeps an unlinked file alive until every handle closes, so
/// a running service will not crash outright; it will keep working from a file
/// that no longer has a name, hold the space we promised to free, and then fail
/// confusingly the next time it tries to load. A download writing into the same
/// directory is worse, because it races.
///
/// So removal is only offered when the owning service is stopped and nothing is
/// downloading or training. Those checks live in the view model, which is what
/// can see service state; this type runs the operation and reports it.

enum ModelRemovalSubject: Equatable {
    /// A directory: a cache repo, a Carey checkpoint, Foundation's weights.
    case directory(URL)
    /// One file inside a cached repo -- a finetune checkpoint, which shares its
    /// repo with other checkpoints someone may still want.
    case repoFile(repo: String, filename: String)
}

struct ModelRemovalPreview: Equatable {
    var subject: ModelRemovalSubject
    var displayName: String
    /// What the directory holds.
    var logicalBytes: Int64
    /// What removing it would actually give back.
    var freeableBytes: Int64
    var files: Int
    /// Cache entries going with it. A Carey checkpoint is a clone of a cache
    /// blob, so removing the checkpoint alone would free nothing; the blobs
    /// that serve only this model are taken too.
    var cacheBlobs: Int

    var sharedBytes: Int64 { max(0, logicalBytes - freeableBytes) }
    var freesNothing: Bool { freeableBytes == 0 && logicalBytes > 0 }
}

struct ModelRemovalResult: Equatable {
    var displayName: String
    var freedBytes: Int64
    var logicalBytes: Int64
}

@MainActor
final class ModelRemovalManager: ObservableObject {
    @Published private(set) var preview: ModelRemovalPreview?
    @Published private(set) var isMeasuring = false
    @Published private(set) var isRemoving = false
    @Published private(set) var lastResult: ModelRemovalResult?
    @Published private(set) var errorMessage: String?

    var isBusy: Bool { isMeasuring || isRemoving }

    func clear() {
        preview = nil
        lastResult = nil
        errorMessage = nil
    }

    func clearError() { errorMessage = nil }

    /// Exactly one notice at a time. A failure and an earlier success were both
    /// held at once, so dismissing the failure uncovered a stale success from a
    /// previous removal and it read as a stack of messages.
    private func report(error: String) {
        lastResult = nil
        errorMessage = error
    }

    private func report(result: ModelRemovalResult) {
        errorMessage = nil
        lastResult = result
    }

    /// Measure first, always. The number shown in the confirmation is the one
    /// the user decides on, so it has to be the honest one rather than the
    /// directory size.
    func measure(
        scriptURL: URL,
        subject: ModelRemovalSubject,
        displayName: String,
        roots: [URL],
        hubDirectory: URL
    ) {
        guard !isBusy else { return }
        isMeasuring = true
        errorMessage = nil
        lastResult = nil
        preview = nil

        let arguments = Self.arguments(
            scriptURL: scriptURL, subject: subject, roots: roots,
            hubDirectory: hubDirectory, dryRun: true
        )
        Task.detached(priority: .userInitiated) {
            let events = Self.run(arguments)
            await MainActor.run {
                self.isMeasuring = false
                if let failure = Self.firstError(in: events) {
                    self.report(error: failure)
                    return
                }
                guard let measured = events.first(where: { $0["event"] as? String == "measured" }) else {
                    self.report(error: "could not measure \(displayName).")
                    return
                }
                self.preview = ModelRemovalPreview(
                    subject: subject,
                    displayName: displayName,
                    logicalBytes: Int64(measured["logical_bytes"] as? Int ?? 0),
                    freeableBytes: Int64(measured["freeable_bytes"] as? Int ?? 0),
                    files: measured["files"] as? Int ?? 0,
                    cacheBlobs: measured["cache_blobs"] as? Int ?? 0
                )
            }
        }
    }

    func remove(
        scriptURL: URL,
        preview: ModelRemovalPreview,
        roots: [URL],
        hubDirectory: URL
    ) {
        guard !isBusy else { return }
        isRemoving = true
        errorMessage = nil
        lastResult = nil

        let arguments = Self.arguments(
            scriptURL: scriptURL, subject: preview.subject, roots: roots,
            hubDirectory: hubDirectory, dryRun: false
        )
        Task.detached(priority: .userInitiated) {
            let events = Self.run(arguments)
            await MainActor.run {
                self.isRemoving = false
                if let failure = Self.firstError(in: events) {
                    self.report(error: failure)
                    return
                }
                guard let done = events.last(where: { $0["event"] as? String == "done" }),
                      done["removed"] as? Bool == true else {
                    self.report(error: "\(preview.displayName) was not removed.")
                    return
                }
                self.report(result: ModelRemovalResult(
                    displayName: preview.displayName,
                    freedBytes: Int64(done["freed_bytes"] as? Int ?? 0),
                    logicalBytes: Int64(done["logical_bytes"] as? Int ?? 0)
                ))
                self.preview = nil
            }
        }
    }

    // MARK: - Process

    nonisolated private static func arguments(
        scriptURL: URL, subject: ModelRemovalSubject, roots: [URL],
        hubDirectory: URL, dryRun: Bool
    ) -> [String] {
        var arguments = ["python3", scriptURL.path]
        for root in roots {
            arguments.append(contentsOf: ["--root", root.path])
        }
        arguments.append(contentsOf: ["--hub", hubDirectory.path])
        switch subject {
        case .directory(let url):
            arguments.append(contentsOf: ["--target", url.path])
        case .repoFile(let repo, let filename):
            arguments.append(contentsOf: ["--repo", repo, "--file", filename])
        }
        if dryRun { arguments.append("--dry-run") }
        return arguments
    }

    nonisolated private static func run(_ arguments: [String]) -> [[String: Any]] {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = arguments
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do { try task.run() } catch { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()

        return String(data: data, encoding: .utf8)?
            .split(separator: "\n")
            .compactMap { line in
                (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any]
            } ?? []
    }

    nonisolated private static func firstError(in events: [[String: Any]]) -> String? {
        events
            .first { $0["event"] as? String == "error" }
            .flatMap { $0["message"] as? String }
    }
}
