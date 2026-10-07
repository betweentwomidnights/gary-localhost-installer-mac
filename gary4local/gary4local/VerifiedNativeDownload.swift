import Foundation
import CryptoKit

nonisolated enum NativeServiceError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

nonisolated private final class NativeDownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let report: @Sendable (Double) -> Void
    let expectedBytes: Int64?
    private let lock = NSLock()
    private var continuation: CheckedContinuation<(URL, URLResponse), Error>?
    private var session: URLSession?
    private var task: URLSessionDownloadTask?
    private var downloadedFile: URL?
    private var wasCancelled = false
    // URLSession's delegate queue is serial; only that queue mutates this value.
    private var lastReport: TimeInterval = 0
    init(expectedBytes: Int64?, report: @escaping @Sendable (Double) -> Void) {
        self.expectedBytes = expectedBytes
        self.report = report
    }
    func download(_ request: URLRequest) async throws -> (URL, URLResponse) {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                guard !wasCancelled else {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                self.continuation = continuation
                let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
                self.session = session
                let task = session.downloadTask(with: request)
                self.task = task
                lock.unlock()
                task.resume()
            }
        } onCancel: { self.cancel() }
    }
    private func cancel() {
        lock.lock()
        wasCancelled = true
        let active = task
        lock.unlock()
        active?.cancel()
    }
    private func finish(_ result: Result<(URL, URLResponse), Error>) {
        lock.lock()
        let waiting = continuation
        continuation = nil
        let session = self.session
        self.session = nil
        task = nil
        let file = downloadedFile
        downloadedFile = nil
        lock.unlock()
        if case .failure = result, let file { try? FileManager.default.removeItem(at: file) }
        session?.finishTasksAndInvalidate()
        waiting?.resume(with: result)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // URLSession deletes location when this callback returns. Move it now,
        // before resuming the caller for off-thread checksum verification.
        let owned = FileManager.default.temporaryDirectory.appendingPathComponent("gary-native-download-\(UUID().uuidString)")
        do {
            try FileManager.default.moveItem(at: location, to: owned)
            lock.lock()
            downloadedFile = owned
            lock.unlock()
        } catch { finish(.failure(error)) }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let file = downloadedFile
        let cancelled = wasCancelled
        lock.unlock()
        if cancelled { finish(.failure(CancellationError())) }
        else if let error { finish(.failure(error)) }
        else if let file, let response = task.response { finish(.success((file, response))) }
        else { finish(.failure(NativeServiceError.invalid("Download completed without a file or HTTP response."))) }
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : expectedBytes ?? 0
        let now = Date.timeIntervalSinceReferenceDate
        if total > 0, now - lastReport >= 0.2 || totalBytesWritten >= total {
            lastReport = now
            report(min(1, Double(totalBytesWritten) / Double(total)))
        }
    }
}

nonisolated enum VerifiedNativeDownload {
    static func hashFile(_ file: URL) async throws -> String {
        let hashing = Task.detached(priority: .userInitiated) { try sha256(file) }
        return try await withTaskCancellationHandler {
            try await hashing.value
        } onCancel: { hashing.cancel() }
    }
    // Stream multi-GB weights; never materialize a whole model in memory.
    static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            try Task.checkCancellation()
            hash.update(data: chunk)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func fetch(_ url: URL, to destination: URL, sha256 expected: String,
                      expectedBytes: Int64? = nil,
                      progress: @escaping @Sendable (Double) -> Void) async throws {
        guard url.scheme == "https", expected.count == 64,
              expected.allSatisfy({ $0.isHexDigit }) else {
            throw NativeServiceError.invalid("A pinned HTTPS download and SHA-256 are required.")
        }
        let delegate = NativeDownloadProgress(expectedBytes: expectedBytes, report: progress)
        var request = URLRequest(url: url)
        request.timeoutInterval = 3600
        // The async convenience download API suppresses download-delegate
        // progress callbacks on macOS. Use a real delegate-backed download task.
        let (temporary, response) = try await delegate.download(request)
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw NativeServiceError.invalid("Download failed: HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0).")
        }
        try Task.checkCancellation()
        let actual = try await hashFile(temporary)
        try Task.checkCancellation()
        guard actual == expected.lowercased() else {
            throw NativeServiceError.invalid("SHA-256 mismatch for \(destination.lastPathComponent); downloaded file was not installed.")
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
    }
}
