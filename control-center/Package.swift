// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "GaryControlCenter",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "GaryControlCenter",
            targets: ["GaryControlCenter"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "GaryControlCenter",
            path: "Sources"
        ),
    ]
)
