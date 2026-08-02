// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ChessVision",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ChessVision", targets: ["ChessVision"])
    ],
    dependencies: [
        .package(url: "https://github.com/chesskit-app/chesskit-swift", exact: "0.9.0")
    ],
    targets: [
        .target(
            name: "ChessVision",
            dependencies: [.product(name: "ChessKit", package: "chesskit-swift")]
        ),
        .testTarget(
            name: "ChessVisionTests",
            dependencies: ["ChessVision"]
        ),
    ]
)
