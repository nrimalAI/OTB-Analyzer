// swift-tools-version: 5.10
import PackageDescription

// A command-line run of the app's own recognizer (Packages/ChessVision, with
// Vision doing the resize as on the phone) over photos with known positions.
// See main.swift.
let package = Package(
    name: "app_check",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../Packages/ChessVision"),
        .package(url: "https://github.com/chesskit-app/chesskit-swift", exact: "0.9.0"),
    ],
    targets: [
        .executableTarget(
            name: "app_check",
            dependencies: [
                "ChessVision",
                .product(name: "ChessKit", package: "chesskit-swift"),
            ]
        )
    ]
)
