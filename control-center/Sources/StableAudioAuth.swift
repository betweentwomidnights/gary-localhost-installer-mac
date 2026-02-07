import Foundation
import Security

enum StableAudioAuthError: LocalizedError {
    case emptyToken
    case keychainFailure(OSStatus)

    var errorDescription: String? {
        switch self {
        case .emptyToken:
            return "Token is empty."
        case .keychainFailure(let status):
            if let message = SecCopyErrorMessageString(status, nil) as String? {
                return "Keychain error: \(message)"
            }
            return "Keychain error (\(status))."
        }
    }
}

enum StableAudioAuthKeychain {
    static let service = "com.betweentwomidnights.gary.localhost.controlcenter"
    static let account = "stable-audio-hf-token"

    static func readToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess else { return nil }
        guard let data = result as? Data else { return nil }
        guard let token = String(data: data, encoding: .utf8), !token.isEmpty else { return nil }
        return token
    }

    static func saveToken(_ token: String) throws {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw StableAudioAuthError.emptyToken
        }

        let data = Data(trimmed.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]

        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
        ]

        let updateStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecSuccess {
            return
        }
        if updateStatus != errSecItemNotFound {
            throw StableAudioAuthError.keychainFailure(updateStatus)
        }

        var addQuery = query
        addQuery[kSecValueData as String] = data
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw StableAudioAuthError.keychainFailure(addStatus)
        }
    }

    static func deleteToken() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]

        let status = SecItemDelete(query as CFDictionary)
        if status == errSecSuccess || status == errSecItemNotFound {
            return
        }
        throw StableAudioAuthError.keychainFailure(status)
    }
}

enum StableAudioAuthLinks {
    static let modelPage = URL(string: "https://huggingface.co/stabilityai/stable-audio-open-small")!
    static let tokenPage = URL(string: "https://huggingface.co/settings/tokens")!
}
