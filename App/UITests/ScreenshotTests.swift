import XCTest

/// Walks the screens and attaches a screenshot of each step to the test
/// result; `.github/workflows/app.yml` exports them for a person to look at.
/// The assertions only check that each step could be taken.
final class ScreenshotTests: XCTestCase {

    override func setUp() {
        continueAfterFailure = false
    }

    private func launch(_ flow: String, environment: [String: String] = [:]) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchEnvironment["AUTOFLOW"] = flow
        app.launchEnvironment.merge(environment) { _, new in new }
        app.launch()
        return app
    }

    /// A ChessVision fixture photo, read from the checkout: the Simulator's
    /// apps can read paths on the Mac.
    private func fixturePhoto(_ name: String) -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "Packages/ChessVision/Tests/ChessVisionTests/Fixtures/\(name)")
            .path
    }

    private func snap(_ name: String) {
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
    }

    /// A board square by its notation; its label starts "g1, ".
    private func square(_ notation: String, in app: XCUIApplication) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", notation + ",")).firstMatch
    }

    private func tap(_ element: XCUIElement, _ what: String) {
        XCTAssertTrue(element.waitForExistence(timeout: 10), "no \(what)")
        element.tap()
    }

    /// The demo scan (`DemoScan`): White's king read as a queen on g1, the
    /// pawn on d4 as Black's.
    func testCorrections() {
        let app = launch("scanned")
        XCTAssertTrue(app.staticTexts["Check the position"].waitForExistence(timeout: 20))
        snap("corrections-1-scanned")

        // "White has no king": Show selects the least sure white queen.
        tap(app.buttons["Show"], "Show")
        snap("corrections-2-show")

        tap(app.buttons["Make it a white king"], "king suggestion")
        snap("corrections-3-king")

        tap(square("d4", in: app), "d4")
        snap("corrections-4-d4")
        tap(app.buttons["Make it a white pawn"], "pawn suggestion")

        tap(app.buttons["Turn the board a quarter"], "quarter turn")
        snap("corrections-5-quarter-turn")
    }

    /// After a scan nobody has said who is to move: the panel asks, and
    /// Analyze asks again before the engine starts.
    func testSideToMove() {
        let app = launch("scanned")
        tap(app.buttons["Show"], "Show")
        tap(app.buttons["Make it a white king"], "king suggestion")
        tap(square("d4", in: app), "d4")
        tap(app.buttons["Make it a white pawn"], "pawn suggestion")
        snap("side-1-question")

        // The Analysis tab asks in a dialog; its keys sit beside the bottom ones.
        tap(app.buttons["Analysis"], "Analysis tab")
        let blackKeys = app.buttons.matching(NSPredicate(format: "label == 'Black to move'"))
        let deadline = Date().addingTimeInterval(5)
        while blackKeys.count < 2, Date() < deadline { usleep(200_000) }
        XCTAssertEqual(blackKeys.count, 2, "no dialog")
        snap("side-2-asked")
        // Dismiss it by tapping outside (newer iOS shows it with no Cancel key).
        app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.08)).tap()
        let closed = Date().addingTimeInterval(5)
        while blackKeys.count > 1, Date() < closed { usleep(200_000) }
        XCTAssertEqual(blackKeys.count, 1, "the dialog did not close")

        // The keys at the bottom answer and analyse in one tap.
        tap(app.buttons["Black to move"], "Black to move")
        // Let the engine report a line or two.
        _ = app.staticTexts["Analyzing\u{2026}"].waitForNonExistence(timeout: 20)
        snap("side-3-analysis")
    }

    /// A photo with no board: what to change, then back to framing.
    func testNoBoard() {
        let app = launch("scan", environment: ["SCAN_PHOTO": "blank"])
        XCTAssertTrue(app.staticTexts["No board found"].waitForExistence(timeout: 30))
        snap("scan-1-no-board")
        // "Retake" with a camera; the Simulator has none, so "Try again".
        tap(app.buttons["Try again"], "Try again")
        XCTAssertTrue(app.buttons["Take photo"].waitForExistence(timeout: 5))
        snap("scan-2-framing")
    }

    /// A real photo through the app's own recognizer, in the Simulator.
    func testRealPhoto() {
        let app = launch("scan", environment: ["SCAN_PHOTO": fixturePhoto("test_000102.jpg")])
        XCTAssertTrue(app.staticTexts["Check the position"].waitForExistence(timeout: 60))
        sleep(1)  // the push from the camera screen
        snap("scan-3-real-photo")
    }

    /// A real photo through to its analysis: the photo beside the heading,
    /// a swipe along the line, and the text export in the share sheet.
    func testAnalysisOfARealPhoto() {
        let app = launch("scan", environment: ["SCAN_PHOTO": fixturePhoto("test_000102.jpg")])
        XCTAssertTrue(app.staticTexts["Check the position"].waitForExistence(timeout: 60))

        tap(app.buttons["Photo"], "Photo")
        XCTAssertTrue(app.buttons["Done"].waitForExistence(timeout: 5))
        snap("analysis-1-photo")
        app.buttons["Done"].tap()

        // After a scan nobody has said who is to move: the bottom key
        // answers it and starts the analysis.
        let white = app.buttons["White to move"]
        XCTAssertTrue(white.waitForExistence(timeout: 5))
        XCTAssertTrue(white.isEnabled, "the fixture should read as a legal position")
        white.tap()
        XCTAssertTrue(app.buttons["Next move"].waitForExistence(timeout: 30))
        // Let the search run on a little, then step one move along the line by swiping.
        sleep(4)
        snap("analysis-2-lines")
        // In Analysis the squares are not buttons; drag leftward across the
        // middle of the board, further than the 40 points a step needs.
        let middle = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label BEGINSWITH 'e4,'")).firstMatch
        XCTAssertTrue(middle.exists, "no square e4 on the board")
        let start = middle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        start.press(forDuration: 0.05, thenDragTo: start.withOffset(CGVector(dx: -150, dy: 0)))
        let stepped = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label CONTAINS[c] 'move 1 of'")).firstMatch
        XCTAssertTrue(stepped.waitForExistence(timeout: 3), "the swipe did not step the line")
        snap("analysis-3-stepped")

        tap(app.buttons["Share"], "Share")
        // The system share sheet, with the text taken when Share was pressed.
        // (Its rows are not always buttons: look for "Copy" in any element.)
        let copy = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Copy'")).firstMatch
        let opened = copy.waitForExistence(timeout: 10)
        snap("analysis-4-share")
        XCTAssertTrue(opened, "no share sheet")
    }

    func testHome() {
        let app = launch("")
        XCTAssertTrue(app.staticTexts["Positions"].waitForExistence(timeout: 20))
        snap("home")
    }
}
