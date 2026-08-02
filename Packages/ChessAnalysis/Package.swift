// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ChessAnalysis",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ChessAnalysis", targets: ["ChessAnalysis"])
    ],
    dependencies: [
        .package(url: "https://github.com/chesskit-app/chesskit-swift", exact: "0.9.0"),
        .package(url: "https://github.com/chesskit-app/chesskit-engine", exact: "0.7.0"),
    ],
    targets: [
        .target(
            name: "ChessAnalysis",
            dependencies: [
                .product(name: "ChessKit", package: "chesskit-swift"),
                .product(name: "ChessKitEngine", package: "chesskit-engine"),
            ]
        ),
        .testTarget(
            name: "ChessAnalysisTests",
            dependencies: ["ChessAnalysis"]
        ),
    ]
)
