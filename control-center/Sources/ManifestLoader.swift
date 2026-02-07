import Foundation

enum ManifestLoaderError: LocalizedError {
    case missingManifest(String)
    case invalidHealthURL(String, serviceID: String)

    var errorDescription: String? {
        switch self {
        case .missingManifest(let path):
            return "Manifest not found at \(path)"
        case .invalidHealthURL(let url, let serviceID):
            return "Invalid health URL '\(url)' for service '\(serviceID)'"
        }
    }
}

struct ManifestLoader {
    static func defaultManifestURL() -> URL {
        if let customPath = ProcessInfo.processInfo.environment["GARY_SERVICE_MANIFEST"], !customPath.isEmpty {
            return URL(fileURLWithPath: customPath)
        }

        var cursor = URL(fileURLWithPath: #filePath)
        while cursor.path != "/" && cursor.lastPathComponent != "Sources" {
            cursor.deleteLastPathComponent()
        }
        if cursor.lastPathComponent == "Sources" {
            cursor.deleteLastPathComponent()
        }
        return cursor.appendingPathComponent("manifest/services.dev.json")
    }

    static func load(from manifestURL: URL) throws -> ResolvedManifest {
        let expandedManifestPath = NSString(string: manifestURL.path).expandingTildeInPath
        let finalManifestURL = URL(fileURLWithPath: expandedManifestPath)

        guard FileManager.default.fileExists(atPath: finalManifestURL.path) else {
            throw ManifestLoaderError.missingManifest(finalManifestURL.path)
        }

        let data = try Data(contentsOf: finalManifestURL)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ServiceManifestFile.self, from: data)

        let manifestDirURL = finalManifestURL.deletingLastPathComponent()
        let workspaceRootURL = manifestDirURL.deletingLastPathComponent().deletingLastPathComponent()
        let appSupportRoot = "\(NSHomeDirectory())/Library/Application Support/GaryLocalhost"

        var variables: [String: String] = [
            "HOME": NSHomeDirectory(),
            "TMPDIR": NSTemporaryDirectory(),
            "MANIFEST_DIR": manifestDirURL.path,
            "WORKSPACE_ROOT": workspaceRootURL.path,
            "APP_SUPPORT_DIR": appSupportRoot,
        ]

        for (key, value) in ProcessInfo.processInfo.environment {
            variables[key] = value
        }
        for (key, value) in decoded.variables {
            variables[key] = expand(value, variables: variables)
        }

        let logDir = URL(fileURLWithPath: normalizePath(expand(decoded.defaultLogDirectory, variables: variables)))

        let resolvedServices = try decoded.services.map { service -> ResolvedService in
            let workingDir = URL(fileURLWithPath: normalizePath(expand(service.workingDirectory, variables: variables)))
            let executable = URL(fileURLWithPath: normalizePath(expand(service.executable, variables: variables)))

            let rawLogPath = expand(service.logFile, variables: variables)
            let logFile: URL
            if rawLogPath.hasPrefix("/") {
                logFile = URL(fileURLWithPath: normalizePath(rawLogPath))
            } else {
                logFile = logDir.appendingPathComponent(rawLogPath)
            }

            let expandedHealthURL = expand(service.healthCheck.url, variables: variables)
            guard let healthURL = URL(string: expandedHealthURL) else {
                throw ManifestLoaderError.invalidHealthURL(expandedHealthURL, serviceID: service.id)
            }

            var resolvedEnv = service.environment
            for (key, value) in service.environment {
                resolvedEnv[key] = expand(value, variables: variables)
            }

            return ResolvedService(
                id: service.id,
                name: service.name,
                workingDirectory: workingDir,
                executable: executable,
                arguments: service.arguments.map { expand($0, variables: variables) },
                environment: resolvedEnv,
                logFile: logFile,
                healthCheck: ResolvedHealthCheck(
                    url: healthURL,
                    expectedStatus: service.healthCheck.expectedStatus,
                    intervalSeconds: max(1, service.healthCheck.intervalSeconds),
                    timeoutSeconds: max(1, service.healthCheck.timeoutSeconds)
                ),
                autoStart: service.autoStart,
                restartOnCrash: service.restartOnCrash,
                gracefulShutdownSeconds: max(1, service.gracefulShutdownSeconds)
            )
        }

        return ResolvedManifest(
            schemaVersion: decoded.schemaVersion,
            appName: decoded.appName,
            defaultLogDirectory: logDir,
            services: resolvedServices
        )
    }

    private static func expand(_ input: String, variables: [String: String]) -> String {
        var output = input
        for _ in 0..<8 {
            var changed = false
            for (key, value) in variables {
                let token = "${\(key)}"
                if output.contains(token) {
                    output = output.replacingOccurrences(of: token, with: value)
                    changed = true
                }
            }
            if !changed { break }
        }
        return output
    }

    private static func normalizePath(_ path: String) -> String {
        let expanded = NSString(string: path).expandingTildeInPath
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }
}
