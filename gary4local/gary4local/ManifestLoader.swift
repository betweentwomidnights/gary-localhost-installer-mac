import Foundation

enum ManifestLoaderError: LocalizedError {
    case missingManifest(String)
    case invalidHealthURL(String, serviceID: String)

    var errorDescription: String? {
        switch self {
        case .missingManifest(let path):
            return "manifest not found at \(path)"
        case .invalidHealthURL(let url, let serviceID):
            return "invalid health url '\(url)' for service '\(serviceID)'"
        }
    }
}

struct ManifestLoader {
    static func defaultManifestURL() -> URL {
        if let customPath = ProcessInfo.processInfo.environment["GARY_SERVICE_MANIFEST"], !customPath.isEmpty {
            return URL(fileURLWithPath: customPath)
        }

        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        let appSupportCandidates = [
            home.appendingPathComponent("Library/Application Support/GaryLocalhost/manifest/services.json"),
            home.appendingPathComponent("Library/Application Support/GaryLocalhost/manifest/services.production.json"),
            home.appendingPathComponent("Library/Application Support/GaryLocalhost/manifest/services.dev.json"),
        ]
        for candidate in appSupportCandidates where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        if let bundledManifest = Bundle.main.url(
            forResource: "services",
            withExtension: "json",
            subdirectory: "manifest"
        ) {
            return bundledManifest
        }
        if let bundledManifest = Bundle.main.url(
            forResource: "services.production",
            withExtension: "json",
            subdirectory: "manifest"
        ) {
            return bundledManifest
        }
        if let bundledManifest = Bundle.main.url(
            forResource: "services.dev",
            withExtension: "json",
            subdirectory: "manifest"
        ) {
            return bundledManifest
        }
        if let bundledManifest = Bundle.main.url(forResource: "services", withExtension: "json") {
            return bundledManifest
        }
        if let bundledManifest = Bundle.main.url(forResource: "services.production", withExtension: "json") {
            return bundledManifest
        }
        if let bundledManifest = Bundle.main.url(forResource: "services.dev", withExtension: "json") {
            return bundledManifest
        }

        let fileManager = FileManager.default
        let cwd = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)

        let candidates = [
            cwd.appendingPathComponent("control-center/manifest/services.dev.json"),
            cwd.appendingPathComponent("manifest/services.dev.json"),
            cwd.appendingPathComponent("../control-center/manifest/services.dev.json"),
            home.appendingPathComponent("gary/gary-localhost-installer-mac/control-center/manifest/services.dev.json"),
            home.appendingPathComponent("gary-localhost-installer-mac/control-center/manifest/services.dev.json"),
        ]

        for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        if let firstCandidate = candidates.first {
            return firstCandidate
        }

        return cwd.appendingPathComponent("control-center/manifest/services.dev.json")
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
        let workspaceRootURL = workspaceRoot(forManifestDirectory: manifestDirURL)
        let appSupportRoot = "\(NSHomeDirectory())/Library/Application Support/GaryLocalhost"
        let bundleResourcesRoot = Bundle.main.resourceURL?.standardizedFileURL.path ?? workspaceRootURL.path
        let appBundleRoot = Bundle.main.bundleURL.standardizedFileURL.path

        var variables: [String: String] = [
            "HOME": NSHomeDirectory(),
            "TMPDIR": NSTemporaryDirectory(),
            "MANIFEST_DIR": manifestDirURL.path,
            "WORKSPACE_ROOT": workspaceRootURL.path,
            "APP_SUPPORT_DIR": appSupportRoot,
            "BUNDLE_RESOURCES": bundleResourcesRoot,
            "APP_BUNDLE_DIR": appBundleRoot,
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

            let resolvedBootstrap: ResolvedBootstrapConfig?
            if let bootstrap = service.bootstrap {
                let venvDirectory = URL(
                    fileURLWithPath: normalizePath(expand(bootstrap.venvDirectory, variables: variables))
                )
                let requirementsFile = URL(
                    fileURLWithPath: normalizePath(expand(bootstrap.requirementsFile, variables: variables))
                )

                resolvedBootstrap = ResolvedBootstrapConfig(
                    pythonExecutable: expand(bootstrap.pythonExecutable, variables: variables),
                    venvDirectory: venvDirectory,
                    requirementsFile: requirementsFile,
                    upgradeBuildTools: bootstrap.upgradeBuildTools,
                    pipArguments: bootstrap.pipArguments.map { expand($0, variables: variables) }
                )
            } else {
                resolvedBootstrap = nil
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
                bootstrap: resolvedBootstrap,
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

    private static func workspaceRoot(forManifestDirectory manifestDirURL: URL) -> URL {
        let normalizedManifestDir = manifestDirURL.standardizedFileURL
        if normalizedManifestDir.lastPathComponent == "manifest" {
            let parent = normalizedManifestDir.deletingLastPathComponent()
            if parent.lastPathComponent == "control-center" {
                return parent.deletingLastPathComponent()
            }
            return parent
        }
        return normalizedManifestDir.deletingLastPathComponent()
    }
}
