import Foundation

struct ServiceManifestFile: Decodable {
    let schemaVersion: Int
    let appName: String
    let variables: [String: String]
    let defaultLogDirectory: String
    let services: [ServiceDefinition]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case appName = "app_name"
        case variables
        case defaultLogDirectory = "default_log_directory"
        case services
    }
}

struct ServiceDefinition: Decodable, Identifiable {
    let id: String
    let name: String
    let workingDirectory: String
    let executable: String
    let arguments: [String]
    let environment: [String: String]
    let logFile: String
    let healthCheck: HealthCheckConfig
    let bootstrap: ServiceBootstrapConfig?
    let autoStart: Bool
    let restartOnCrash: Bool
    let gracefulShutdownSeconds: Int

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case workingDirectory = "working_directory"
        case executable
        case arguments
        case environment
        case logFile = "log_file"
        case healthCheck = "health_check"
        case bootstrap
        case autoStart = "auto_start"
        case restartOnCrash = "restart_on_crash"
        case gracefulShutdownSeconds = "graceful_shutdown_seconds"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        workingDirectory = try container.decode(String.self, forKey: .workingDirectory)
        executable = try container.decode(String.self, forKey: .executable)
        arguments = try container.decodeIfPresent([String].self, forKey: .arguments) ?? []
        environment = try container.decodeIfPresent([String: String].self, forKey: .environment) ?? [:]
        logFile = try container.decodeIfPresent(String.self, forKey: .logFile) ?? "\(id).log"
        healthCheck = try container.decode(HealthCheckConfig.self, forKey: .healthCheck)
        bootstrap = try container.decodeIfPresent(ServiceBootstrapConfig.self, forKey: .bootstrap)
        autoStart = try container.decodeIfPresent(Bool.self, forKey: .autoStart) ?? false
        restartOnCrash = try container.decodeIfPresent(Bool.self, forKey: .restartOnCrash) ?? false
        gracefulShutdownSeconds = try container.decodeIfPresent(Int.self, forKey: .gracefulShutdownSeconds) ?? 8
    }
}

struct ServiceBootstrapConfig: Decodable {
    let pythonExecutable: String
    let venvDirectory: String
    let requirementsFile: String
    let upgradeBuildTools: Bool
    let pipArguments: [String]

    enum CodingKeys: String, CodingKey {
        case pythonExecutable = "python_executable"
        case venvDirectory = "venv_directory"
        case requirementsFile = "requirements_file"
        case upgradeBuildTools = "upgrade_build_tools"
        case pipArguments = "pip_arguments"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        pythonExecutable = try container.decodeIfPresent(String.self, forKey: .pythonExecutable) ?? "python3"
        venvDirectory = try container.decode(String.self, forKey: .venvDirectory)
        requirementsFile = try container.decode(String.self, forKey: .requirementsFile)
        upgradeBuildTools = try container.decodeIfPresent(Bool.self, forKey: .upgradeBuildTools) ?? true
        pipArguments = try container.decodeIfPresent([String].self, forKey: .pipArguments) ?? []
    }
}

struct HealthCheckConfig: Decodable {
    let url: String
    let expectedStatus: Int
    let intervalSeconds: Int
    let timeoutSeconds: Int

    enum CodingKeys: String, CodingKey {
        case url
        case expectedStatus = "expected_status"
        case intervalSeconds = "interval_seconds"
        case timeoutSeconds = "timeout_seconds"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        url = try container.decode(String.self, forKey: .url)
        expectedStatus = try container.decodeIfPresent(Int.self, forKey: .expectedStatus) ?? 200
        intervalSeconds = try container.decodeIfPresent(Int.self, forKey: .intervalSeconds) ?? 3
        timeoutSeconds = try container.decodeIfPresent(Int.self, forKey: .timeoutSeconds) ?? 2
    }
}

struct ResolvedManifest {
    let schemaVersion: Int
    let appName: String
    let defaultLogDirectory: URL
    let services: [ResolvedService]
}

struct ResolvedService: Identifiable {
    let id: String
    let name: String
    let workingDirectory: URL
    let executable: URL
    let arguments: [String]
    let environment: [String: String]
    let logFile: URL
    let healthCheck: ResolvedHealthCheck
    let bootstrap: ResolvedBootstrapConfig?
    let autoStart: Bool
    let restartOnCrash: Bool
    let gracefulShutdownSeconds: Int
}

struct ResolvedHealthCheck {
    let url: URL
    let expectedStatus: Int
    let intervalSeconds: Int
    let timeoutSeconds: Int
}

struct ResolvedBootstrapConfig {
    let pythonExecutable: String
    let venvDirectory: URL
    let requirementsFile: URL
    let upgradeBuildTools: Bool
    let pipArguments: [String]
}

enum ProcessState: String {
    case stopped
    case starting
    case running
    case stopping
    case failed
}

enum HealthState: String {
    case unknown
    case healthy
    case unhealthy
}

enum BootstrapState: String {
    case notConfigured
    case ready
    case running
    case succeeded
    case failed
}

struct ServiceRuntime: Identifiable {
    let service: ResolvedService
    var processState: ProcessState = .stopped
    var healthState: HealthState = .unknown
    var bootstrapState: BootstrapState
    var bootstrapMessage: String?
    var pid: Int32?
    var lastExitCode: Int32?
    var lastError: String?

    init(service: ResolvedService) {
        self.service = service
        self.bootstrapState = service.bootstrap == nil ? .notConfigured : .ready
    }

    var id: String { service.id }
    var isRunning: Bool {
        processState == .starting || processState == .running || processState == .stopping
    }
    var isBootstrapping: Bool {
        bootstrapState == .running
    }
}
