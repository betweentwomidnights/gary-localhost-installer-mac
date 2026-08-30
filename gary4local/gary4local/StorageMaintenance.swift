import Combine
import Foundation

/// Sizes of the things gary4local puts on disk, and the duplicate-space pass.
///
/// The install is mostly weights, and the weights were being stored twice: the
/// downloader fetched into the Hugging Face cache and then copied the file into
/// the checkpoints tree. The downloader clones now, but copies already on disk
/// stay until this runs.
///
/// This is not the same duplication the Windows build reclaims. There the hub
/// cannot symlink without developer mode and writes each file twice inside one
/// repo cache. macOS symlinks fine, so that never happens here.

struct StorageUsage: Equatable {
    var huggingFaceCacheBytes: Int64 = 0
    var careyCheckpointsBytes: Int64 = 0
    var environmentsBytes: Int64 = 0
    var otherBytes: Int64 = 0

    var totalBytes: Int64 {
        huggingFaceCacheBytes + careyCheckpointsBytes + environmentsBytes + otherBytes
    }
}

struct StorageReclaimResult: Equatable {
    var wasDryRun: Bool
    var matchedBytes: Int64
    var reclaimedFiles: Int
    var skipped: Int
    var freeDeltaBytes: Int64
}

@MainActor
final class StorageMaintenanceManager: ObservableObject {
    @Published private(set) var usage: StorageUsage?
    @Published private(set) var isScanning = false
    @Published private(set) var isReclaiming = false
    @Published private(set) var progressMessage = ""
    @Published private(set) var lastResult: StorageReclaimResult?
    @Published private(set) var errorMessage: String?

    private var process: Process?

    var isBusy: Bool { isScanning || isReclaiming }

    func clearError() { errorMessage = nil }

    // MARK: - Sizing

    /// Directory sizes, measured off the main thread. A full install is
    /// hundreds of thousands of files, so this is slow enough to freeze the UI
    /// if it runs inline.
    func scan(appSupportDirectory: URL, checkpointsDirectory: URL?) {
        guard !isBusy else { return }
        isScanning = true
        errorMessage = nil

        let hub = appSupportDirectory
            .appendingPathComponent("cache/huggingface", isDirectory: true)
        let envs = appSupportDirectory.appendingPathComponent("venvs", isDirectory: true)
        let checkpoints = checkpointsDirectory

        Task.detached(priority: .utility) {
            let hubBytes = Self.directorySize(hub)
            let envBytes = Self.directorySize(envs)
            let checkpointBytes = checkpoints.map(Self.directorySize) ?? 0
            let totalBytes = Self.directorySize(appSupportDirectory)
            // Checkpoints usually live inside app support; don't count twice.
            let counted = hubBytes + envBytes
                + (Self.isInside(checkpoints, appSupportDirectory) ? checkpointBytes : 0)
            let other = max(0, totalBytes - counted)

            let usage = StorageUsage(
                huggingFaceCacheBytes: hubBytes,
                careyCheckpointsBytes: checkpointBytes,
                environmentsBytes: envBytes,
                otherBytes: other
            )
            await MainActor.run {
                self.usage = usage
                self.isScanning = false
            }
        }
    }

    nonisolated private static func isInside(_ path: URL?, _ root: URL) -> Bool {
        guard let path else { return false }
        return path.standardizedFileURL.path.hasPrefix(root.standardizedFileURL.path)
    }

    nonisolated private static func directorySize(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        var seenInodes = Set<UInt64>()
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: Set(keys)),
                  values.isRegularFile == true else { continue }
            // A clone reports its full size on both sides, and hard links would
            // be counted once per link. Count each inode once so the number
            // matches what removing the tree would actually give back.
            if let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               let inode = attrs[.systemFileNumber] as? UInt64 {
                if seenInodes.contains(inode) { continue }
                seenInodes.insert(inode)
            }
            let size = values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0
            total += Int64(size)
        }
        return total
    }

    // MARK: - Reclaim

    func reclaim(
        scriptURL: URL,
        checkpointsDirectory: URL,
        huggingFaceHubDirectory: URL,
        dryRun: Bool
    ) {
        guard !isBusy else { return }
        guard FileManager.default.isReadableFile(atPath: scriptURL.path) else {
            errorMessage = "the reclaim helper is missing from the runtime."
            return
        }
        isReclaiming = true
        errorMessage = nil
        lastResult = nil
        progressMessage = dryRun ? "looking for duplicated files..." : "reclaiming duplicated files..."

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        var arguments = [
            "python3",
            scriptURL.path,
            "--checkpoints-dir", checkpointsDirectory.path,
            "--hf-cache", huggingFaceHubDirectory.path,
        ]
        if dryRun { arguments.append("--dry-run") }
        task.arguments = arguments

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()
        process = task

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            for line in text.split(separator: "\n") {
                guard let payload = try? JSONSerialization.jsonObject(
                    with: Data(line.utf8)
                ) as? [String: Any] else { continue }
                Task { @MainActor [weak self] in
                    self?.handle(event: payload)
                }
            }
        }

        task.terminationHandler = { [weak self] finished in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.isReclaiming = false
                self.process = nil
                if finished.terminationStatus != 0, self.errorMessage == nil {
                    self.errorMessage = "the reclaim pass exited with status \(finished.terminationStatus)."
                }
                self.progressMessage = ""
            }
        }

        do {
            try task.run()
        } catch {
            isReclaiming = false
            process = nil
            errorMessage = error.localizedDescription
        }
    }

    private func handle(event: [String: Any]) {
        let name = event["event"] as? String ?? ""
        switch name {
        case "start":
            let candidates = event["candidate_files"] as? Int ?? 0
            progressMessage = "checking \(candidates) file\(candidates == 1 ? "" : "s")..."
        case "progress":
            if let file = event["file"] as? String,
               let index = event["index"] as? Int,
               let total = event["total"] as? Int {
                progressMessage = "\(index)/\(total) — \(file)"
            }
        case "error":
            errorMessage = event["message"] as? String ?? "the reclaim pass failed."
        case "done":
            lastResult = StorageReclaimResult(
                wasDryRun: event["dry_run"] as? Bool ?? false,
                matchedBytes: Int64(event["matched_bytes"] as? Int ?? 0),
                reclaimedFiles: event["reclaimed_files"] as? Int ?? 0,
                skipped: event["skipped"] as? Int ?? 0,
                freeDeltaBytes: Int64(event["free_delta_bytes"] as? Int ?? 0)
            )
            progressMessage = ""
        default:
            break
        }
    }

    func cancel() {
        process?.terminate()
        process = nil
        isReclaiming = false
        progressMessage = ""
    }
}
