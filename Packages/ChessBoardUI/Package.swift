// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ChessBoardUI",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ChessBoardUI", targets: ["ChessBoardUI"])
    ],
    dependencies: [
        .package(url: "https://github.com/chesskit-app/chesskit-swift", exact: "0.9.0")
    ],
    targets: [
        .target(
            name: "ChessBoardUI",
            dependencies: [.product(name: "ChessKit", package: "chesskit-swift")]
        ),
        .testTarget(
            name: "ChessBoardUITests",
            dependencies: ["ChessBoardUI"]
        ),
    ]
)
